"""Serialize supported model payloads independently of an execution framework."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def _copy_path(value: object, destination: Path, label: str) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} payload does not exist: {source}")
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
    return True


def write_onnx(model: object, destination: str | Path) -> int:
    """Write ONNX bytes, a path-backed model, or a protobuf-like model."""

    selected = Path(destination)
    if _copy_path(model, selected, "ONNX"):
        return selected.stat().st_size
    if isinstance(model, bytes):
        selected.write_bytes(model)
        return len(model)
    serialize = getattr(model, "SerializeToString", None)
    if callable(serialize):
        value = serialize()
        if isinstance(value, bytes):
            selected.write_bytes(value)
            return len(value)
    raise TypeError(
        "model must be ONNX bytes, an ONNX path, or expose SerializeToString()"
    )


def write_joblib(model: object, destination: str | Path) -> int:
    """Write an existing Joblib payload or serialize a fitted predictor."""

    selected = Path(destination)
    if _copy_path(model, selected, "Joblib"):
        return selected.stat().st_size
    try:
        import joblib
    except ImportError as error:
        raise ImportError("Joblib is required to serialize this model") from error
    if not callable(getattr(model, "predict", None)) and not callable(model):
        raise TypeError("Joblib models must define predict() or be callable")
    if type(model).__module__ in {"__main__", "__mp_main__"}:
        import cloudpickle

        with selected.open("wb") as stream:
            cloudpickle.dump(model, stream, protocol=5)
    else:
        joblib.dump(model, selected, compress=0, protocol=5)
    return selected.stat().st_size


def equinox_leaves(model: object) -> list[tuple[str, str, list[int], int, int]]:
    """Return the stable array-leaf layout stored in an Equinox manifest."""

    try:
        import jax
    except ImportError as error:
        raise ImportError("JAX is required to serialize an Equinox model") from error
    result = []
    offset = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        shape = getattr(leaf, "shape", None)
        dtype_value = getattr(leaf, "dtype", None)
        if shape is None or dtype_value is None or not shape:
            continue
        dtype = str(dtype_value)
        if dtype not in {"float32", "float64", "int32", "int64"}:
            continue
        count = int(getattr(leaf, "size")) * int(getattr(dtype_value, "itemsize"))
        if count:
            result.append((jax.tree_util.keystr(path), dtype, list(shape), offset, count))
            offset += count
    if not result:
        raise ValueError("Equinox model must contain at least one supported array leaf")
    return result


def write_equinox(model: object, destination: str | Path, *, batched: bool) -> tuple[int, list[tuple[str, str, list[int], int, int]]]:
    """Serialize one callable Equinox PyTree and return its leaf layout."""

    if not callable(model):
        raise TypeError("Equinox models must be callable")
    try:
        import cloudpickle
        import equinox as _  # noqa: F401
    except ImportError as error:
        raise ImportError("Equinox is required to serialize this model") from error
    leaves = equinox_leaves(model)
    selected = Path(destination)
    with selected.open("wb") as stream:
        cloudpickle.dump(
            {"schema": "bindnordic.equinox", "model": model, "batched": batched},
            stream,
            protocol=5,
        )
    return selected.stat().st_size, leaves


__all__ = ["equinox_leaves", "write_equinox", "write_joblib", "write_onnx"]
