"""Decode stored OpenFOAM volume fields and their metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


class OpenFOAMDecodeError(ValueError):
    """Raised when a field does not use the supported OpenFOAM representation."""


@dataclass(frozen=True, slots=True)
class FoamField:
    path: Path
    name: str
    field_class: str
    dimensions: str | None
    values: np.ndarray
    uniform: bool
    boundaries: tuple[str, ...]


_COMMENTS = re.compile(rb"/\*.*?\*/|//[^\n]*", re.DOTALL)
_COMPONENTS = {b"scalar": 1, b"vector": 3, b"symmTensor": 6, b"tensor": 9}
_NONUNIFORM = re.compile(
    rb"\b(?:internalField\s+)?nonuniform\s+"
    rb"List<(scalar|vector|symmTensor|tensor)>\s+(\d+)\s*\("
)


def _block(payload: bytes, name: bytes) -> bytes | None:
    match = re.search(rb"\b" + re.escape(name) + rb"\s*\{", payload)
    if match is None:
        return None
    start = match.end()
    depth = 1
    for index in range(start, len(payload)):
        if payload[index] == ord("{"):
            depth += 1
        elif payload[index] == ord("}"):
            depth -= 1
            if depth == 0:
                return payload[start:index]
    raise OpenFOAMDecodeError(f"unterminated {name.decode()} dictionary")


def _entry(block: bytes, name: bytes) -> str | None:
    match = re.search(
        rb"\b" + re.escape(name) + rb'\s+(?:"([^"]*)"|([^;]+));', block
    )
    if match is None:
        return None
    value = match.group(1) if match.group(1) is not None else match.group(2)
    return value.decode().strip()


def _boundary_names(payload: bytes) -> tuple[str, ...]:
    value = _block(payload, b"boundaryField")
    if value is None:
        return ()
    names = []
    depth = 0
    token = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char == ord("{"):
            if depth == 0:
                name = token.decode().strip().split()
                if name:
                    names.append(name[-1])
            depth += 1
            token.clear()
        elif char == ord("}"):
            depth -= 1
            token.clear()
        elif depth == 0:
            token.append(char)
        index += 1
    return tuple(names)


def _uniform_value(payload: bytes) -> np.ndarray | None:
    match = re.search(rb"\binternalField\s+uniform\s+([^;]+);", payload, re.DOTALL)
    if match is None:
        return None
    values = np.fromstring(match.group(1).replace(b"(", b" ").replace(b")", b" "), sep=" ")
    if values.size not in {1, 3, 6, 9}:
        raise OpenFOAMDecodeError("uniform internalField has an invalid component count")
    return values


def _binary_dtypes(arch: str | None) -> tuple[np.dtype, ...]:
    if arch is None:
        return (np.dtype("<f8"), np.dtype("<f4"))
    byte_order = "<" if arch.startswith("LSB;") else ">" if arch.startswith("MSB;") else "="
    match = re.search(r"(?:^|;)scalar=(32|64)(?:;|$)", arch)
    if match is None:
        return (np.dtype(f"{byte_order}f8"), np.dtype(f"{byte_order}f4"))
    return (np.dtype(f"{byte_order}f{int(match.group(1)) // 8}"),)


def _binary_region(
    payload: bytes,
    start: int,
    count: int,
    components: int,
    arch: str | None,
) -> tuple[int, int, np.dtype]:
    starts = (start, start + 1) if payload[start : start + 1] == b"\n" else (start,)
    if payload[start : start + 2] == b"\r\n":
        starts = (start, start + 2)
    expected = []
    for dtype in _binary_dtypes(arch):
        size = count * components * dtype.itemsize
        expected.append(size)
        for data_start in starts:
            data_end = data_start + size
            if re.match(rb"\s*\)\s*;", payload[data_end:]) is not None:
                return data_start, data_end, dtype
    values = " or ".join(str(size) for size in expected)
    raise OpenFOAMDecodeError(
        f"binary field byte count is invalid: expected {values} bytes before its terminator"
    )


def _binary_payload(
    payload: bytes,
    start: int,
    count: int,
    components: int,
    arch: str | None,
) -> np.ndarray:
    data_start, data_end, dtype = _binary_region(
        payload, start, count, components, arch
    )
    return np.frombuffer(payload[data_start:data_end], dtype=dtype).astype(
        dtype.newbyteorder("="), copy=True
    )


def _structural_payload(payload: bytes, binary: bool, arch: str | None) -> bytes:
    """Remove binary field bytes before parsing OpenFOAM dictionaries."""

    if not binary:
        return _COMMENTS.sub(b" ", payload)
    clean = bytearray(payload)
    for match in _NONUNIFORM.finditer(payload):
        kind, raw_count = match.groups()
        data_start, data_end, _ = _binary_region(
            payload,
            match.end(),
            int(raw_count),
            _COMPONENTS[kind],
            arch,
        )
        clean[data_start:data_end] = b" " * (data_end - data_start)
    return _COMMENTS.sub(b" ", bytes(clean))


def _nonuniform_value(payload: bytes, binary: bool, arch: str | None) -> np.ndarray:
    match = re.search(
        rb"\binternalField\s+" + _NONUNIFORM.pattern,
        payload,
    )
    if match is None:
        raise OpenFOAMDecodeError("could not locate a supported internalField")
    kind, raw_count = match.groups()
    count = int(raw_count)
    components = _COMPONENTS[kind]
    start = match.end()
    if binary:
        values = _binary_payload(payload, start, count, components, arch)
    else:
        end = re.search(rb"\)\s*;", payload[start:], re.DOTALL)
        if end is None:
            raise OpenFOAMDecodeError("ASCII internalField is unterminated")
        raw = payload[start : start + end.start()]
        values = np.fromstring(raw.replace(b"(", b" ").replace(b")", b" "), sep=" ")
        if values.size != count * components:
            raise OpenFOAMDecodeError("ASCII internalField count does not match its declaration")
    return values if components == 1 else values.reshape(count, components)


def read_field(path: str | Path, *, cell_count: int | None = None) -> FoamField:
    """Read one ASCII or native-endian float OpenFOAM volume field."""

    selected = Path(path).expanduser().resolve()
    raw = selected.read_bytes()
    header_payload = _COMMENTS.sub(b" ", raw)
    header = _block(header_payload, b"FoamFile")
    if header is None:
        raise OpenFOAMDecodeError("field is missing its FoamFile dictionary")
    field_class = _entry(header, b"class")
    name = _entry(header, b"object") or selected.name
    if not field_class:
        raise OpenFOAMDecodeError("field is missing its OpenFOAM class")
    format_name = _entry(header, b"format") or "ascii"
    arch = _entry(header, b"arch")
    clean = _structural_payload(raw, format_name == "binary", arch)
    values = _uniform_value(raw)
    uniform = values is not None
    if values is None:
        values = _nonuniform_value(raw, format_name == "binary", arch)
    elif cell_count is not None:
        values = np.broadcast_to(values, (cell_count, values.size)).copy()
        if values.shape[1] == 1:
            values = values[:, 0]
    elif values.size == 1:
        values = values[0]
    dimensions = _entry(clean, b"dimensions")
    return FoamField(
        selected,
        name,
        field_class,
        dimensions,
        np.asarray(values),
        uniform,
        _boundary_names(clean),
    )
