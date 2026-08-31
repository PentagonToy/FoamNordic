"""Managed Python model worker behind the native Fjord exchange boundary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

from ..random import Key, invocation, key as random_key
from ..core.layout import FieldLayout

try:
    from .. import _native
except ImportError:
    _native = None


_MODEL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_JOBLIB_MMAP_THRESHOLD = 16 * 1024 * 1024


@dataclass(frozen=True)
class _EmbeddedPayload:
    path: Path
    offset: int
    size: int


def _configure_model_threads(threads: int) -> int:
    """Apply one bounded CPU budget before importing model runtimes."""

    threads = int(threads)
    if threads < 1:
        raise ValueError("model threads must be positive")
    for name in _MODEL_THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    preserved_xla_flags = tuple(
        value
        for value in os.environ.get("XLA_FLAGS", "").split()
        if "xla_cpu_multi_thread_eigen" not in value
        and "intra_op_parallelism_threads" not in value
    )
    os.environ["XLA_FLAGS"] = " ".join(
        (
            *preserved_xla_flags,
            "--xla_cpu_multi_thread_eigen=true",
            f"intra_op_parallelism_threads={threads}",
        )
    )
    return threads


def _configure_estimator_threads(model, threads: int) -> None:
    """Set n_jobs throughout an already-fitted scikit-learn model graph."""

    pending = [model]
    visited: set[int] = set()
    while pending:
        estimator = pending.pop()
        if estimator is None or isinstance(estimator, (str, bytes)):
            continue
        identity = id(estimator)
        if identity in visited:
            continue
        visited.add(identity)
        if hasattr(estimator, "n_jobs"):
            try:
                setattr(estimator, "n_jobs", threads)
            except (AttributeError, TypeError):
                pass
        for name in (
            "estimators_",
            "estimators",
            "named_estimators_",
            "estimator_",
            "estimator",
            "steps",
            "named_steps",
            "transformers_",
            "transformers",
        ):
            children = getattr(estimator, name, None)
            if children is None:
                continue
            if isinstance(children, Mapping):
                pending.extend(children.values())
            elif isinstance(children, Sequence) and not isinstance(
                children, (str, bytes)
            ):
                for child in children:
                    pending.append(
                        child[1]
                        if isinstance(child, tuple) and len(child) == 2
                        else child
                    )
            else:
                pending.append(children)


def _manifest(path: Path) -> dict[str, object]:
    return dict(_native.read_model_manifest(str(path)))


def _payload(manifest_path: Path, manifest: Mapping[str, object]):
    if bool(manifest.get("bundled", False)):
        if (
            str(manifest.get("format")) == "joblib"
            and os.name == "posix"
            and manifest_path.stat().st_size >= _JOBLIB_MMAP_THRESHOLD
        ):
            offset, size = _native.read_model_payload_region(str(manifest_path))
            return _EmbeddedPayload(manifest_path, int(offset), int(size))
        return BytesIO(_native.read_model_payload(str(manifest_path)))
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


def _fd_payload_path(stream, offset: int) -> str | None:
    """Return a descriptor path only when reopening preserves its offset."""

    candidates = (
        (f"/proc/self/fd/{stream.fileno()}", f"/dev/fd/{stream.fileno()}")
        if sys.platform.startswith("linux")
        else (f"/dev/fd/{stream.fileno()}", f"/proc/self/fd/{stream.fileno()}")
    )
    for candidate in candidates:
        try:
            stream.seek(offset)
            descriptor = os.open(candidate, os.O_RDONLY)
            try:
                reopened_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
            finally:
                os.close(descriptor)
            stream.seek(offset)
        except OSError:
            continue
        if reopened_offset == offset:
            return candidate
    return None


def _staged_joblib(path: _EmbeddedPayload, joblib):
    storage = tempfile.TemporaryDirectory(prefix="foamnordic-joblib-")
    load_path = Path(storage.name) / "model.joblib"
    _native.extract_model_payload(str(path.path), str(load_path))
    try:
        model = joblib.load(load_path, mmap_mode="r")
    except ValueError:
        # Dynamically packaged callables are ordinary cloudpickle streams and
        # do not provide a memory-mappable array payload.
        model = joblib.load(load_path)
    return model, storage, "staged"


def _load_joblib(path, joblib):
    if not isinstance(path, _EmbeddedPayload):
        if isinstance(path, Path):
            try:
                return joblib.load(path, mmap_mode="r"), None, "file"
            except ValueError:
                return joblib.load(path), None, "file"
        return joblib.load(path), None, "buffered"

    # FNOBND2 places the standalone Joblib stream at a 64-byte boundary.  Keep
    # the bundle descriptor open while Joblib follows its descriptor path so
    # NumPy can map array pages directly from the single FNOM file.
    if path.offset % 64 == 0:
        with path.path.open("rb") as stream:
            descriptor_path = _fd_payload_path(stream, path.offset)
            if descriptor_path is not None:
                stream.seek(path.offset)
                try:
                    return joblib.load(descriptor_path, mmap_mode="r"), None, "direct"
                except Exception:
                    # Descriptor reopening differs across kernels and filesystems.
                    # The streamed fallback preserves correctness and mmap use.
                    pass
    return _staged_joblib(path, joblib)


def _joblib_evaluator(
    path: Path,
    outputs,
    runtime: str = "sklearn",
    threads: int = 1,
):
    threads = _configure_model_threads(threads)
    _activate_joblib_runtime(runtime)
    try:
        import joblib
        import numpy as np
    except ImportError as error:
        raise ImportError("Joblib is missing from this FoamNordic installation") from error
    model, payload_storage, payload_mode = _load_joblib(path, joblib)
    if isinstance(model, dict) and model.get("schema") == "foamnordic.function/v1":
        return _function_evaluator(model, outputs)
    _configure_estimator_threads(model, threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limiter = threadpool_limits(limits=threads)
    except ImportError:
        threadpool_limiter = None
    predict = getattr(model, "predict", None)
    function = predict if callable(predict) else model
    if not callable(function):
        raise TypeError("Joblib payload must define predict() or be callable")

    def evaluate(
        buffer,
        rows,
        columns,
        dtype,
        _exchange_index,
        _physical_time,
        _rank=0,
        _threadpool_limiter=threadpool_limiter,
        _payload_storage=payload_storage,
        _payload_mode=payload_mode,
    ):
        # Retain the process-wide threadpoolctl limiter for the evaluator's
        # lifetime.  Native BLAS/OpenMP pools may already have been loaded by
        # the embedding Python process before this module starts.
        del _threadpool_limiter, _payload_storage, _payload_mode
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        return _packed_result(function(features), outputs, rows, dtype)

    evaluate.foamnordic_payload_mode = payload_mode
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
    accepts_seed = "seed" in parameters and "seed" not in input_names
    accepts_exchange_index = (
        "exchange_index" in parameters and "exchange_index" not in input_names
    )
    accepts_physical_time = (
        "physical_time" in parameters and "physical_time" not in input_names
    )
    accepts_rank = "rank" in parameters and "rank" not in input_names
    accepts_rng = "rng" in parameters and "rng" not in input_names
    accepts_key = "key" in parameters and "key" not in input_names

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
        if accepts_seed:
            arguments["seed"] = root_key.entropy[0]
        if accepts_exchange_index:
            arguments["exchange_index"] = exchange_index
        if accepts_physical_time:
            arguments["physical_time"] = float(physical_time)
        if accepts_rank:
            arguments["rank"] = rank
        if accepts_rng or accepts_key:
            call_key = invocation(root_key, program, exchange_index, rank)
            if accepts_rng:
                arguments["rng"] = np.random.default_rng(
                    np.random.SeedSequence(
                        call_key.entropy,
                        spawn_key=call_key.path,
                    )
                )
            if accepts_key:
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
    key: Key | int | None = None,
    program: str = "model",
    *,
    seed: int | None = None,
    threads: int = 1,
):
    _configure_model_threads(threads)
    try:
        import cloudpickle
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError as error:
        raise ImportError("Equinox is missing from this FoamNordic installation") from error
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    if isinstance(path, Path):
        with path.open("rb") as stream:
            package = cloudpickle.load(stream)
    else:
        path.seek(0)
        package = cloudpickle.load(path)
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
    parser.add_argument("--threads", type=int, default=1)
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
    model_threads = _configure_model_threads(arguments.threads)
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = _manifest(manifest_path)
    model_format = str(manifest["format"])
    payload = _payload(manifest_path, manifest)
    outputs = _output_contract(manifest)
    dtype = str(manifest["inputs"][0][2])
    if model_format == "joblib":
        runtime = str(manifest.get("runtime") or "sklearn")
        evaluator = _joblib_evaluator(payload, outputs, runtime, model_threads)
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
