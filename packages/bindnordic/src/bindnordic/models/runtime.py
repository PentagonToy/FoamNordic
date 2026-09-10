"""Load and evaluate portable Python model payloads without solver semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import errno
import gc
from io import BytesIO
import inspect
import os
from pathlib import Path
import tempfile

from .artifact import extract_payload, read_payload
from .compiler import compile_source


_MODEL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class EmbeddedPayload:
    path: Path
    offset: int
    size: int


def configure_model_threads(threads: int) -> int:
    """Apply one bounded CPU budget before importing model runtimes."""

    threads = int(threads)
    if threads < 1:
        raise ValueError("model threads must be positive")
    for name in _MODEL_THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    preserved = tuple(
        value
        for value in os.environ.get("XLA_FLAGS", "").split()
        if "xla_cpu_multi_thread_eigen" not in value
        and "intra_op_parallelism_threads" not in value
    )
    os.environ["XLA_FLAGS"] = " ".join(
        (
            *preserved,
            "--xla_cpu_multi_thread_eigen=true",
            f"intra_op_parallelism_threads={threads}",
        )
    )
    return threads


def _configure_estimator_threads(model: object, threads: int) -> None:
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


def packed_result(value: object, outputs, rows: int, dtype: str):
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
            "this artifact requires scikit-learn-intelex on a supported "
            "Linux x86-64 worker"
        ) from error
    patch_sklearn(verbose=False)


def _direct_embedded_joblib(path: EmbeddedPayload, joblib):
    numpy_pickle = getattr(joblib, "numpy_pickle", None)
    unpickle = getattr(numpy_pickle, "_unpickle", None)
    if not callable(unpickle):
        return None
    try:
        parameters = inspect.signature(unpickle).parameters
    except (TypeError, ValueError):
        return None
    if not {"fobj", "ensure_native_byte_order", "filename", "mmap_mode"}.issubset(
        parameters
    ):
        return None
    try:
        with path.path.open("rb") as stream:
            stream.seek(path.offset)
            return unpickle(
                stream,
                ensure_native_byte_order=False,
                filename=os.fspath(path.path),
                mmap_mode="r",
            )
    except (OSError, TypeError, ValueError):
        return None


def _load_joblib_file(path: Path, joblib):
    try:
        return joblib.load(path, mmap_mode="r"), True
    except ValueError:
        pass
    except OSError as error:
        if error.errno not in {errno.EMFILE, errno.ENFILE}:
            raise
    gc.collect()
    return joblib.load(path), False


def _staged_joblib(path: EmbeddedPayload, joblib):
    storage = tempfile.TemporaryDirectory(prefix="bindnordic-joblib-")
    load_path = Path(storage.name) / "model.joblib"
    extract_payload(path.path, load_path)
    model, mapped = _load_joblib_file(load_path, joblib)
    if mapped:
        return model, storage, "staged"
    storage.cleanup()
    return model, None, "buffered"


def _load_joblib(path, joblib):
    if not isinstance(path, EmbeddedPayload):
        if isinstance(path, Path):
            model, mapped = _load_joblib_file(path, joblib)
            return model, None, "file" if mapped else "buffered"
        return joblib.load(path), None, "buffered"
    if path.offset % 64 == 0:
        model = _direct_embedded_joblib(path, joblib)
        if model is not None:
            return model, None, "direct"
        gc.collect()
    return _staged_joblib(path, joblib)


def joblib_evaluator(
    path,
    outputs,
    runtime: str = "sklearn",
    threads: int = 1,
    *,
    object_evaluator: Callable[[object], Callable[..., object] | None] | None = None,
):
    threads = configure_model_threads(threads)
    _activate_joblib_runtime(runtime)
    try:
        import joblib
        import numpy as np
    except ImportError as error:
        raise ImportError("Joblib is missing from this BindNordic installation") from error
    model, payload_storage, payload_mode = _load_joblib(path, joblib)
    if object_evaluator is not None:
        specialized = object_evaluator(model)
        if specialized is not None:
            def evaluate_specialized(*args, _storage=payload_storage, **kwargs):
                del _storage
                return specialized(*args, **kwargs)

            evaluate_specialized.bindnordic_payload_mode = payload_mode
            return evaluate_specialized
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
        del _threadpool_limiter, _payload_storage, _payload_mode
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        return packed_result(function(features), outputs, rows, dtype)

    evaluate.bindnordic_payload_mode = payload_mode
    return evaluate


def compiled_evaluator(path, outputs, dtype: str, threads: int = 1):
    import numpy as np

    if dtype != "float64":
        raise ValueError("compiled artifacts require float64 tensors")
    if sum(width for _, width in outputs) != 1:
        raise ValueError("compiled artifacts require one scalar output")
    if isinstance(path, BytesIO):
        source = path.getvalue()
    elif isinstance(path, Path):
        source = path.read_bytes()
    elif isinstance(path, EmbeddedPayload):
        source = read_payload(path.path)
    else:
        source = bytes(path)
    library_path, cache_hit, compile_seconds = compile_source(source)
    library = ctypes.CDLL(str(library_path))

    def symbol(current: str, legacy: str):
        try:
            return getattr(library, current)
        except AttributeError:
            return getattr(library, legacy)

    input_width = symbol("bindnordic_input_width", "foamnordic_input_width")
    input_width.argtypes = []
    input_width.restype = ctypes.c_size_t
    output_width = symbol("bindnordic_output_width", "foamnordic_output_width")
    output_width.argtypes = []
    output_width.restype = ctypes.c_size_t
    predict = symbol("bindnordic_predict", "foamnordic_predict")
    predict.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    predict.restype = ctypes.c_int
    expected_inputs = int(input_width())
    expected_outputs = int(output_width())
    try:
        predict_parallel = symbol(
            "bindnordic_predict_parallel", "foamnordic_predict_parallel"
        )
    except AttributeError:
        predict_parallel = None
    if predict_parallel is not None:
        predict_parallel.argtypes = [*predict.argtypes, ctypes.c_size_t]
        predict_parallel.restype = ctypes.c_int
    pool = (
        ThreadPoolExecutor(max_workers=threads)
        if threads > 1 and predict_parallel is None
        else None
    )

    def call(features, result, row_start: int, row_stop: int) -> None:
        count = row_stop - row_start
        input_address = features.ctypes.data + row_start * features.strides[0]
        output_address = result.ctypes.data + row_start * result.strides[0]
        status = predict(
            ctypes.cast(input_address, ctypes.POINTER(ctypes.c_double)),
            count,
            expected_inputs,
            ctypes.cast(output_address, ctypes.POINTER(ctypes.c_double)),
            expected_outputs,
        )
        if status != 0:
            raise RuntimeError(f"compiled FNOM inference failed with status {status}")

    def evaluate(
        buffer,
        rows,
        columns,
        value_dtype,
        _exchange_index,
        _physical_time,
        _rank=0,
        _library=library,
        _pool=pool,
    ):
        del _library
        if value_dtype != "float64" or columns != expected_inputs:
            raise ValueError("compiled FNOM input does not match its native ABI")
        features = np.frombuffer(buffer, dtype=np.float64).reshape(rows, columns)
        if not features.flags.c_contiguous:
            features = np.ascontiguousarray(features)
        result = np.empty((rows, expected_outputs), dtype=np.float64)
        workers = min(threads, rows)
        if predict_parallel is not None:
            status = predict_parallel(
                features.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                rows,
                columns,
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                expected_outputs,
                workers,
            )
            if status != 0:
                raise RuntimeError(f"compiled FNOM inference failed with status {status}")
        elif _pool is None or workers <= 1:
            call(features, result, 0, rows)
        else:
            quotient, remainder = divmod(rows, workers)
            futures = []
            start = 0
            for worker in range(workers):
                stop = start + quotient + (worker < remainder)
                futures.append(_pool.submit(call, features, result, start, stop))
                start = stop
            for future in futures:
                future.result()
        return result

    metadata = {
        "cache_hit": cache_hit,
        "compile_seconds": compile_seconds,
        "library": library_path,
        "threads": threads,
        "parallel": "native" if predict_parallel is not None else "python",
    }
    for name, value in metadata.items():
        setattr(evaluate, f"bindnordic_{name}", value)
    return evaluate


def equinox_evaluator(
    path,
    outputs,
    dtype: str,
    *,
    key_factory: Callable[[int, int], object] | None = None,
    threads: int = 1,
):
    configure_model_threads(threads)
    try:
        import cloudpickle
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError as error:
        raise ImportError("Equinox is missing from this BindNordic installation") from error
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    if isinstance(path, Path):
        with path.open("rb") as stream:
            package = cloudpickle.load(stream)
    else:
        path.seek(0)
        package = cloudpickle.load(path)
    if not isinstance(package, dict) or package.get("schema") not in {
        "bindnordic.equinox",
        "foamnordic.equinox",
    }:
        raise ValueError("unsupported Equinox payload schema")
    model = package["model"]
    try:
        accepts_key = "key" in inspect.signature(model).parameters
    except (TypeError, ValueError):
        accepts_key = False
    if accepts_key and key_factory is None:
        raise ValueError("keyed Equinox models require an invocation key factory")
    batched = bool(package.get("batched", False))
    if accepts_key:
        function = (
            (lambda values, key: model(values, key=key))
            if batched
            else jax.vmap(lambda value, key: model(value, key=key), in_axes=(0, 0))
        )
    else:
        function = model if batched else jax.vmap(model)
    compiled = jax.jit(function)

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
            call_key = key_factory(int(exchange_index), int(rank))
            keys = call_key if batched else jax.random.split(call_key, rows)
            result = compiled(values, keys)
        else:
            result = compiled(values)
        return packed_result(result, outputs, rows, value_dtype)

    return evaluate
