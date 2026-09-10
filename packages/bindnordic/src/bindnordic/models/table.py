"""Portable regular-grid table payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import struct

import numpy as np


_HEADER = struct.Struct("<8sBBHQ")
_AXIS = struct.Struct("<BQ")
_MAGIC = b"FNOTAB1\0"


def write_table(
    path: str | Path,
    axes: Mapping[str, Sequence[float] | np.ndarray],
    values: np.ndarray,
    output_features: int,
    *,
    dtype: str = "float64",
    bounds: str | Mapping[str, str] = "clamp",
) -> int:
    """Write a deterministic dense N-dimensional table payload."""

    if dtype not in {"float32", "float64"}:
        raise ValueError("table dtype must be float32 or float64")
    if not axes or len(axes) > 8:
        raise ValueError("tables require between one and eight axes")
    if output_features < 1:
        raise ValueError("table output feature count must be positive")

    points = []
    policies = []
    for name, raw_axis in axes.items():
        axis = np.asarray(raw_axis, dtype=np.float64)
        if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
            raise ValueError(f"table axis {name!r} must contain at least two finite points")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"table axis {name!r} must be strictly increasing")
        policy = bounds[name] if isinstance(bounds, Mapping) else bounds
        if policy not in {"clamp", "error"}:
            raise ValueError(f"table axis {name!r} bounds must be clamp or error")
        points.append(np.ascontiguousarray(axis))
        policies.append(policy)

    table = np.asarray(values)
    expected = tuple(axis.size for axis in points)
    if output_features == 1 and table.shape == expected:
        table = table[..., np.newaxis]
    if table.shape != (*expected, output_features):
        raise ValueError(
            f"table values must have shape {(*expected, output_features)!r}"
        )
    if not np.all(np.isfinite(table)):
        raise ValueError("table values must be finite")
    table = np.ascontiguousarray(
        table, dtype="<f4" if dtype == "float32" else "<f8"
    )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(
            _HEADER.pack(
                _MAGIC,
                1 if dtype == "float32" else 2,
                len(points),
                0,
                output_features,
            )
        )
        for axis, policy in zip(points, policies, strict=True):
            stream.write(_AXIS.pack(0 if policy == "clamp" else 1, axis.size))
            stream.write(axis.astype("<f8", copy=False).tobytes())
        stream.write(table.tobytes())
    return destination.stat().st_size


__all__ = ["write_table"]
