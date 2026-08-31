"""Managed Python model worker behind the native Fjord exchange boundary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import inspect
import json
from pathlib import Path
import sys

from ..random import Key, invocation, key as random_key
from ..core.layout import FieldLayout

try:
    from .. import _native
except ImportError:
    _native = None


def _manifest(path: Path) -> dict[str, object]:
    return dict(_native.read_model_manifest(str(path)))


def _payload(manifest_path: Path, manifest: Mapping[str, object]) -> Path:
    value = Path(str(manifest["artifact_path"]))
    return value if value.is_absolute() else manifest_path.parent / value


def _output_contract(manifest: Mapping[str, object]) -> tuple[tuple[str, int], ...]:
    return tuple((str(item[0]), int(item[1])) for item in manifest["outputs"])


def _packed_result(value, outputs, rows: int, dtype: str):
    import numpy as np

    if isinstance(value, Mapping):
        pieces = [np.asarray(value[name]).reshape(rows, width) for name, width in outputs]
        result = np.concatenate(pieces, axis=1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(outputs):
            raise ValueError("model output count does not match the FNOM contract")
        pieces = [
            np.asarray(item).reshape(rows, width)
            for item, (_, width) in zip(value, outputs, strict=True)
        ]
        result = np.concatenate(pieces, axis=1)
    else:
        result = np.asarray(value).reshape(rows, sum(width for _, width in outputs))
    return np.ascontiguousarray(result, dtype=dtype)


def _physical_input(values, layout: Mapping[str, object], rows: int):
    """Decode a packed OpenFOAM port into its Python physical shape."""

    return FieldLayout.from_plan(layout).unpack(values, rows)


def _transport_output(values, layout: Mapping[str, object], rows: int):
    """Encode one Python physical value into its OpenFOAM packed shape."""

    return FieldLayout.from_plan(layout).pack(values, rows)


def _packed_function_result(value, outputs, layouts, rows: int, dtype: str):
    """Pack logical function outputs using their OpenFOAM value layouts."""

    import numpy as np

    if len(layouts) != len(outputs):
        raise ValueError("Operator.function output layouts do not match its ports")
    if isinstance(value, Mapping):
        values = tuple(value[name] for name, _ in outputs)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(outputs):
            raise ValueError("model output count does not match the FNOM contract")
        values = tuple(value)
    else:
        values = (value,)
        if len(outputs) != 1:
            raise ValueError("model output count does not match the FNOM contract")
    pieces = [
        _transport_output(item, layout, rows)
        for item, layout in zip(values, layouts, strict=True)
    ]
    return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=dtype)


def _activate_joblib_runtime(runtime: str) -> None:
    if runtime == "sklearn":
        return
    if runtime != "sklearnex":
        raise ValueError(f"unsupported Joblib runtime {runtime!r}")
    try:
        from sklearnex import patch_sklearn
    except ImportError as error:
        raise ImportError(
            "this artifact requires scikit-learn-intelex; install it on a "
            "supported Linux x86-64 worker"
        ) from error
    patch_sklearn(verbose=False)


def _joblib_evaluator(path: Path, outputs, runtime: str = "sklearn"):
    _activate_joblib_runtime(runtime)
    try:
        import joblib
        import numpy as np
    except ImportError as error:
        raise ImportError("Joblib is missing from this FoamNordic installation") from error
    try:
        model = joblib.load(path, mmap_mode="r")
    except ValueError:
        # Dynamically packaged callables are ordinary cloudpickle streams and
        # do not provide a memory-mappable array payload.
        model = joblib.load(path)
    if isinstance(model, dict) and model.get("schema") == "foamnordic.function/v1":
        return _function_evaluator(model, outputs)
    predict = getattr(model, "predict", None)
    function = predict if callable(predict) else model
    if not callable(function):
        raise TypeError("Joblib payload must define predict() or be callable")

    def evaluate(
        buffer, rows, columns, dtype, _exchange_index, _physical_time, _rank=0
    ):
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        return _packed_result(function(features), outputs, rows, dtype)

    return evaluate


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
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}

    root_key = (
        Key.from_plan(package["key"])
        if "key" in package
        else random_key(int(package.get("seed", 42)))
    )
    program = str(package.get("program", "function"))

    def evaluate(
        buffer, rows, columns, dtype, exchange_index, physical_time, rank=0
    ):
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        widths = package.get("input_widths")
        layouts = package.get("input_layouts")
        if widths is None:
            # Widths are copied into the package by the launcher. Retain a
            # clear diagnostic for artifacts created by an older launcher.
            raise ValueError("Operator.function payload is missing input widths")
        arguments = {}
        offset = 0
        if layouts is not None and len(layouts) != len(input_names):
            raise ValueError("Operator.function input layouts do not match its ports")
        for index, (name, width) in enumerate(
            zip(input_names, widths, strict=True)
        ):
            width = int(width)
            value = features[:, offset : offset + width]
            arguments[name] = (
                _physical_input(value, layouts[index], rows)
                if layouts is not None
                else value[:, 0]
                if width == 1
                else value
            )
            offset += width
        if offset != columns:
            raise ValueError("Operator.function input widths do not match the buffer")

        call_key = invocation(root_key, program, int(exchange_index), int(rank))
        optional = {
            "seed": root_key.entropy[0],
            "exchange_index": int(exchange_index),
            "physical_time": float(physical_time),
            "rank": int(rank),
            "rng": np.random.default_rng(
                np.random.SeedSequence(call_key.entropy, spawn_key=call_key.path)
            ),
            "key": call_key,
        }
        for name, value in optional.items():
            if name in parameters and name not in arguments:
                arguments[name] = value
        result = function(**arguments)
        output_layouts = package.get("output_layouts")
        if output_layouts is None:
            return _packed_result(result, outputs, rows, dtype)
        return _packed_function_result(
            result, outputs, output_layouts, rows, dtype
        )

    return evaluate


def _equinox_evaluator(
    path: Path,
    outputs,
    dtype: str,
    key: Key | int | None = None,
    program: str = "model",
    *,
    seed: int | None = None,
):
    try:
        import cloudpickle
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError as error:
        raise ImportError("Equinox is missing from this FoamNordic installation") from error
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    with path.open("rb") as stream:
        package = cloudpickle.load(stream)
    if not isinstance(package, dict) or package.get("schema") != "foamnordic.equinox/v1":
        raise ValueError("unsupported Equinox payload schema")
    model = package["model"]
    try:
        accepts_key = "key" in inspect.signature(model).parameters
    except (TypeError, ValueError):
        accepts_key = False
    batched = bool(package.get("batched", False))
    if accepts_key:
        if batched:
            function = lambda values, key: model(values, key=key)
        else:
            function = jax.vmap(
                lambda value, key: model(value, key=key),
                in_axes=(0, 0),
            )
    else:
        function = model if batched else jax.vmap(model)
    compiled = jax.jit(function)
    if key is not None and seed is not None:
        raise ValueError("Equinox evaluator accepts either key or seed, not both")
    if seed is not None:
        key = seed
    root_key = (
        random_key(42)
        if key is None
        else random_key(key)
        if isinstance(key, int)
        else key
    )

    def evaluate(
        buffer,
        rows,
        columns,
        value_dtype,
        exchange_index,
        _physical_time,
        rank=0,
    ):
        features = np.frombuffer(buffer, dtype=value_dtype).reshape(rows, columns)
        values = jnp.asarray(features)
        if accepts_key:
            from ..random import to_jax

            call_key = to_jax(
                invocation(root_key, program, int(exchange_index), int(rank))
            )
            keys = call_key if batched else jax.random.split(call_key, rows)
            result = compiled(values, keys)
        else:
            result = compiled(values)
        return _packed_result(result, outputs, rows, value_dtype)

    return evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foamnordic.execution.resident",
        description="Run a Joblib or Equinox model behind native Fjord transport.",
    )
    parser.add_argument("address")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--connections", type=int, default=1)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--key", default=Key((42, 0)).to_json())
    parser.add_argument("--program", default="model")
    parser.add_argument("--no-shm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if _native is None:
        raise RuntimeError(
            "the managed resident requires a FoamNordic binary wheel"
        )
    arguments = _parser().parse_args(argv)
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = _manifest(manifest_path)
    model_format = str(manifest["format"])
    payload = _payload(manifest_path, manifest)
    outputs = _output_contract(manifest)
    dtype = str(manifest["inputs"][0][2])
    if model_format == "joblib":
        runtime = str(manifest.get("runtime") or "sklearn")
        evaluator = _joblib_evaluator(payload, outputs, runtime)
    elif model_format == "equinox":
        evaluator = _equinox_evaluator(
            payload,
            outputs,
            dtype,
            Key.from_plan(json.loads(arguments.key)),
            arguments.program,
        )
    else:
        raise ValueError(f"managed Python worker does not support {model_format!r}")
    runtime_label = (
        f" ({manifest['runtime']})" if manifest.get("runtime") is not None else ""
    )
    print(
        f"[FoamNordic] {model_format.title()} model{runtime_label} loaded once: "
        f"{payload}"
    )
    _native.run_python_worker(
        arguments.address,
        str(manifest_path),
        evaluator,
        arguments.connections,
        arguments.ready_file,
        not arguments.no_shm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
