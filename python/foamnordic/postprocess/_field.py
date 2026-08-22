"""OpenFOAM time discovery and durable internal-field decoding."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import math
import re

import numpy as np


MESH_FILES = ("points", "faces", "owner", "neighbour")


def processor_directories(case_dir: Path) -> tuple[Path, ...]:
    processors = sorted(
        (
            path
            for path in case_dir.glob("processor*")
            if path.is_dir() and path.name.removeprefix("processor").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("processor")),
    )
    indices = [int(path.name.removeprefix("processor")) for path in processors]
    if indices and indices != list(range(len(processors))):
        raise ValueError(
            "OpenFOAM processor directories must be consecutively numbered from processor0"
        )
    return tuple(processors)


def numeric_times(directory: Path) -> dict[float, str]:
    result: dict[float, str] = {}
    for path in directory.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if value in result:
            raise ValueError(
                f"OpenFOAM case {directory} contains duplicate physical time {value:g}"
            )
        result[value] = path.name
    return result


def available_times(case_dir: Path) -> tuple[float, ...]:
    processors = processor_directories(case_dir)
    if not processors:
        return tuple(sorted(numeric_times(case_dir)))
    values = set(numeric_times(processors[0]))
    for processor in processors[1:]:
        values.intersection_update(numeric_times(processor))
    return tuple(sorted(values))


def select_time(
    case_dir: Path,
    *,
    time_idx: int | None,
    physical_time: float | None,
) -> float:
    if time_idx is not None and physical_time is not None:
        raise ValueError("time_idx and physical_time are mutually exclusive")
    times = available_times(case_dir)
    if not times:
        raise FileNotFoundError(f"No OpenFOAM time directories were found in {case_dir}")
    if physical_time is not None:
        if isinstance(physical_time, bool) or not isinstance(physical_time, (int, float)):
            raise TypeError("physical_time must be a finite number")
        requested = float(physical_time)
        if not math.isfinite(requested):
            raise ValueError("physical_time must be finite")
        for value in times:
            if math.isclose(value, requested, rel_tol=1.0e-12, abs_tol=1.0e-14):
                return value
        available = ", ".join(f"{value:g}" for value in times)
        raise KeyError(
            f"physical_time {requested:g} was not found; available times: {available}"
        )
    index = -1 if time_idx is None else time_idx
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("time_idx must be an integer")
    try:
        return times[index]
    except IndexError as error:
        raise IndexError(
            f"time_idx {index} is out of range; found {len(times)} stored times"
        ) from error


def _field_paths(case_dir: Path, physical_time: float, field_name: str) -> tuple[Path, ...]:
    if not field_name or Path(field_name).name != field_name:
        raise ValueError("field name must be one non-empty path component")
    direct_times = numeric_times(case_dir)
    if physical_time in direct_times:
        direct = case_dir / direct_times[physical_time] / field_name
        if direct.is_file():
            return (direct,)
    processors = processor_directories(case_dir)
    if not processors:
        raise FileNotFoundError(
            f"OpenFOAM field {field_name!r} was not found at time {physical_time:g}"
        )
    paths = []
    for processor in processors:
        times = numeric_times(processor)
        if physical_time not in times:
            raise FileNotFoundError(
                f"{processor.name} does not contain physical time {physical_time:g}"
            )
        path = processor / times[physical_time] / field_name
        if not path.is_file():
            raise FileNotFoundError(
                f"OpenFOAM field {field_name!r} is missing from {processor.name} "
                f"at time {physical_time:g}"
            )
        paths.append(path)
    return tuple(paths)


def _read_ascii_internal_field(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    format_match = re.search(rb"\bformat\s+(ascii|binary)\s*;", payload)
    if format_match is not None and format_match.group(1) != b"ascii":
        raise ValueError(f"Cannot use the ASCII fallback for binary field {path}")
    uniform = re.search(rb"\binternalField\s+uniform\s+([^;]+);", payload, re.DOTALL)
    if uniform is not None:
        values = np.fromstring(
            uniform.group(1).replace(b"(", b" ").replace(b")", b" "), sep=" "
        )
        return values[0] if values.size == 1 else values
    nonuniform = re.search(
        rb"\binternalField\s+nonuniform\s+List<[^>]+>\s+(\d+)\s*\((.*?)\)\s*;",
        payload,
        re.DOTALL,
    )
    if nonuniform is None:
        raise ValueError(f"Cannot locate a supported internalField in {path}")
    count = int(nonuniform.group(1))
    values = np.fromstring(
        nonuniform.group(2).replace(b"(", b" ").replace(b")", b" "), sep=" "
    )
    if count == 0 or values.size % count:
        raise ValueError(f"Cannot reshape internalField values from {path}")
    components = values.size // count
    return values if components == 1 else values.reshape(count, components)


def _cell_count(field_path: Path) -> int:
    owner = field_path.parent.parent / "constant/polyMesh/owner"
    if not owner.is_file():
        raise ValueError(
            f"Cannot expand uniform OpenFOAM field because mesh owner file is missing: {owner}"
        )
    match = re.search(rb"\bnCells\s*:\s*(\d+)\b", owner.read_bytes()[:65536])
    if match is None:
        raise ValueError(f"Mesh cell count is missing from {owner}")
    return int(match.group(1))


def read_field_file(path: Path) -> np.ndarray:
    try:
        from foamlib import FoamFieldFile, FoamFileDecodeError
    except ImportError as error:
        raise ImportError("FoamNordic postprocessing requires foamlib") from error
    payload = path.read_bytes()
    try:
        values = np.asarray(FoamFieldFile(path).internal_field)
    except FoamFileDecodeError:
        values = np.asarray(_read_ascii_internal_field(path))
    if re.search(rb"\binternalField\s+uniform\b", payload):
        values = np.broadcast_to(values, (_cell_count(path), *values.shape)).copy()
    return values


def read_case_field(case_dir: Path, field_name: str, physical_time: float) -> np.ndarray:
    arrays = [
        read_field_file(path)
        for path in _field_paths(case_dir, physical_time, field_name)
    ]
    if len(arrays) == 1:
        return arrays[0]
    shape = arrays[0].shape[1:]
    if any(array.shape[1:] != shape for array in arrays[1:]):
        raise ValueError(f"Decomposed field {field_name!r} has incompatible shapes")
    return np.concatenate(arrays, axis=0)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()
