"""Managed Python model worker behind the native Fjord exchange boundary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import sys

from bindnordic.models.artifact import load as load_artifact
from bindnordic.models.artifact import payload_region, read_payload
from bindnordic.models.runtime import EmbeddedPayload as _EmbeddedPayload
from bindnordic.models.runtime import compiled_evaluator as _compiled_evaluator
from bindnordic.models.runtime import configure_model_threads as _configure_model_threads
from bindnordic.models.runtime import equinox_evaluator as _bind_equinox_evaluator
from bindnordic.models.runtime import joblib_evaluator as _bind_joblib_evaluator
from bindnordic.models.runtime import packed_result as _packed_result

from ..random import Key, invocation, key as random_key
from ..core.layout import FieldLayout

try:
    from .. import _native
except ImportError:
    _native = None


_JOBLIB_MMAP_THRESHOLD = 16 * 1024 * 1024


def _manifest(path: Path) -> dict[str, object]:
    return load_artifact(path).as_dict()


def _payload(manifest_path: Path, manifest: Mapping[str, object]):
    if bool(manifest.get("bundled", False)):
        if (
            str(manifest.get("format")) == "joblib"
            and os.name == "posix"
            and manifest_path.stat().st_size >= _JOBLIB_MMAP_THRESHOLD
        ):
            offset, size = payload_region(manifest_path)
            return _EmbeddedPayload(manifest_path, int(offset), int(size))
        return BytesIO(read_payload(manifest_path))
    value = Path(str(manifest["artifact_path"]))
    return value if value.is_absolute() else manifest_path.parent / value


def _output_contract(manifest: Mapping[str, object]) -> tuple[tuple[str, int], ...]:
    return tuple((str(item[0]), int(item[1])) for item in manifest["outputs"])


def _joblib_evaluator(
    path,
    outputs,
    runtime: str = "sklearn",
    threads: int = 1,
    *,
    input_names: tuple[str, ...] | None = None,
    input_format: str = "numpy",
):
    def specialized(model):
        if isinstance(model, dict) and model.get("schema") == "foamnordic.function":
            return _function_evaluator(model, outputs)
        return None

    return _bind_joblib_evaluator(
        path,
        outputs,
        runtime,
        threads,
        input_names=input_names,
        input_format=input_format,
        object_evaluator=specialized,
    )


def _function_evaluator(package, outputs):
    """Build a deterministic logical-port evaluator for Operator.function."""

    import numpy as np

    function = package.get("function")
    if not callable(function):
        raise TypeError("Operator.function payload does not contain a callable")
    input_names = tuple(str(value) for value in package.get("inputs", ()))
    output_names = tuple(str(value) for value in package.get("outputs", ()))
    if not input_names or not output_names:
        raise ValueError("Operator.function payload has an empty tensor contract")
    raw_widths = package.get("input_widths")
    if raw_widths is None:
        raise ValueError("Operator.function payload is missing input widths")
    widths = tuple(int(width) for width in raw_widths)
    if len(widths) != len(input_names) or any(width <= 0 for width in widths):
        raise ValueError("Operator.function input widths do not match its ports")
    raw_input_layouts = package.get("input_layouts")
    input_layouts = (
        None
        if raw_input_layouts is None
        else tuple(FieldLayout.from_plan(layout) for layout in raw_input_layouts)
    )
    if input_layouts is not None and len(input_layouts) != len(input_names):
        raise ValueError("Operator.function input layouts do not match its ports")
    raw_output_layouts = package.get("output_layouts")
    output_layouts = (
        None
        if raw_output_layouts is None
        else tuple(FieldLayout.from_plan(layout) for layout in raw_output_layouts)
    )
    if output_layouts is not None and len(output_layouts) != len(outputs):
        raise ValueError("Operator.function output layouts do not match its ports")
    input_ports = []
    offset = 0
    for index, (name, width) in enumerate(zip(input_names, widths, strict=True)):
        input_ports.append(
            (
                name,
                slice(offset, offset + width),
                None if input_layouts is None else input_layouts[index],
                width,
            )
        )
        offset += width
    input_ports = tuple(input_ports)
    input_columns = offset
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_exchange_index = (
        "exchange_index" in parameters and "exchange_index" not in input_names
    )
    accepts_physical_time = (
        "physical_time" in parameters and "physical_time" not in input_names
    )
    accepts_rank = "rank" in parameters and "rank" not in input_names
    accepts_key = "key" in parameters and "key" not in input_names

    root_key = Key.from_plan(package["key"])
    program = str(package.get("program", "function"))

    def evaluate(
        buffer, rows, columns, dtype, exchange_index, physical_time, rank=0
    ):
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        if columns != input_columns:
            raise ValueError("Operator.function input widths do not match the buffer")
        arguments = {}
        for name, column_slice, layout, width in input_ports:
            value = features[:, column_slice]
            arguments[name] = (
                layout.unpack(value, rows)
                if layout is not None
                else value[:, 0]
                if width == 1
                else value
            )

        exchange_index = int(exchange_index)
        rank = int(rank)
        if accepts_exchange_index:
            arguments["exchange_index"] = exchange_index
        if accepts_physical_time:
            arguments["physical_time"] = float(physical_time)
        if accepts_rank:
            arguments["rank"] = rank
        if accepts_key:
            call_key = invocation(root_key, program, exchange_index, rank)
            arguments["key"] = call_key
        result = function(**arguments)
        if output_layouts is None:
            packed = _packed_result(result, outputs, rows, dtype)
        else:
            if isinstance(result, Mapping):
                values = tuple(result[name] for name, _ in outputs)
            elif isinstance(result, Sequence) and not isinstance(
                result, (str, bytes)
            ):
                if len(result) != len(outputs):
                    raise ValueError(
                        "model output count does not match the FNOM contract"
                    )
                values = tuple(result)
            else:
                values = (result,)
                if len(outputs) != 1:
                    raise ValueError(
                        "model output count does not match the FNOM contract"
                    )
            pieces = [
                layout.pack(value, rows)
                for value, layout in zip(values, output_layouts, strict=True)
            ]
            packed = np.ascontiguousarray(
                np.concatenate(pieces, axis=1),
                dtype=dtype,
            )
        return packed

    return evaluate


def _equinox_evaluator(
    path,
    outputs,
    dtype: str,
    key: Key | None = None,
    program: str = "model",
    *,
    threads: int = 1,
):
    from ..random import to_jax

    root_key = random_key(42) if key is None else key

    def key_factory(exchange_index: int, rank: int):
        return to_jax(invocation(root_key, program, exchange_index, rank))

    return _bind_equinox_evaluator(
        path,
        outputs,
        dtype,
        key_factory=key_factory,
        threads=threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foamnordic.execution.resident",
        description="Run a compiled, Joblib, or Equinox model behind native Fjord transport.",
    )
    parser.add_argument("address")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--connections", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--receive-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--profile-every", type=int, default=0)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--key", default=Key((42, 0)).to_json())
    parser.add_argument("--program", default="model")
    parser.add_argument("--no-shm", action="store_true")
    return parser


def run_program(
    argv: list[str], capacity=None, *, configure_threads: bool = True
) -> int:
    if _native is None:
        raise RuntimeError(
            "the managed resident requires a FoamNordic binary wheel"
        )
    arguments = _parser().parse_args(argv)
    if arguments.profile_every < 0:
        raise ValueError("profile_every must not be negative")
    if arguments.receive_bytes <= 0:
        raise ValueError("receive_bytes must be positive")
    model_threads = (
        _configure_model_threads(arguments.threads)
        if configure_threads
        else arguments.threads
    )
    if capacity is None:
        capacity = _native._ExecutionCapacity(model_threads)
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = _manifest(manifest_path)
    model_format = str(manifest["format"])
    if model_format in {"onnx", "table"} or manifest.get("runtime") == "forest":
        _native.run_native_worker(
            arguments.address, str(manifest_path), arguments.connections,
            arguments.ready_file, not arguments.no_shm, model_threads,
            arguments.receive_bytes, arguments.profile_every, arguments.program,
            capacity,
        )
        return 0
    payload = _payload(manifest_path, manifest)
    if model_format == "compiled":
        from ..build.compiler import compile_source

        source = (
            payload.read_bytes() if isinstance(payload, Path)
            else read_payload(payload.path)
            if isinstance(payload, _EmbeddedPayload)
            else payload.getvalue()
        )
        library, _, _ = compile_source(source)
        _native.run_compiled_worker(
            arguments.address, str(manifest_path), str(library),
            arguments.connections, arguments.ready_file, not arguments.no_shm,
            model_threads, arguments.receive_bytes, arguments.profile_every,
            arguments.program,
            capacity,
        )
        return 0
    outputs = _output_contract(manifest)
    dtype = str(manifest["inputs"][0][2])
    if model_format == "joblib":
        runtime = str(manifest.get("runtime") or "sklearn")
        evaluator = _joblib_evaluator(
            payload,
            outputs,
            runtime,
            model_threads,
            input_names=tuple(str(item[0]) for item in manifest["inputs"]),
            input_format=str(manifest.get("input_format") or "numpy"),
        )
    elif model_format == "equinox":
        evaluator = _equinox_evaluator(
            payload,
            outputs,
            dtype,
            Key.from_plan(json.loads(arguments.key)),
            arguments.program,
            threads=model_threads,
        )
    else:
        raise ValueError(f"managed Python worker does not support {model_format!r}")
    runtime_label = (
        f" ({manifest['runtime']})" if manifest.get("runtime") is not None else ""
    )
    payload_label = "embedded payload" if bool(manifest.get("bundled", False)) else payload
    print(
        f"[FoamNordic] {model_format.title()} model{runtime_label} loaded once: "
        f"{payload_label}"
    )
    print(f"[FoamNordic] Model CPU budget: {model_threads}")
    _native.run_python_worker(
        arguments.address,
        str(manifest_path),
        evaluator,
        arguments.connections,
        arguments.ready_file,
        not arguments.no_shm,
        model_threads,
        arguments.receive_bytes,
        arguments.profile_every,
        arguments.program,
        capacity,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_program(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
