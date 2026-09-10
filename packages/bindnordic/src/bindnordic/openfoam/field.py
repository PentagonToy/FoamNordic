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
    match = re.search(rb"\b" + re.escape(name) + rb"\s+([^;]+);", block)
    return None if match is None else match.group(1).decode().strip()


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


def _nonuniform_value(payload: bytes, binary: bool) -> np.ndarray:
    match = re.search(
        rb"\binternalField\s+nonuniform\s+List<(scalar|vector|symmTensor|tensor)>\s+(\d+)\s*\(",
        payload,
    )
    if match is None:
        raise OpenFOAMDecodeError("could not locate a supported internalField")
    kind, raw_count = match.groups()
    count = int(raw_count)
    components = _COMPONENTS[kind]
    start = match.end()
    if binary:
        end_marker = payload.find(b")", start)
        if end_marker < 0:
            raise OpenFOAMDecodeError("binary internalField is unterminated")
        raw = payload[start:end_marker]
        expected64 = count * components * 8
        expected32 = count * components * 4
        if len(raw) == expected64:
            values = np.frombuffer(raw, dtype=np.dtype("<f8")).copy()
        elif len(raw) == expected32:
            values = np.frombuffer(raw, dtype=np.dtype("<f4")).copy()
        else:
            raise OpenFOAMDecodeError(
                f"binary internalField byte count is invalid: expected {expected64} or {expected32}, got {len(raw)}"
            )
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
    clean = _COMMENTS.sub(b" ", raw)
    header = _block(clean, b"FoamFile")
    if header is None:
        raise OpenFOAMDecodeError("field is missing its FoamFile dictionary")
    field_class = _entry(header, b"class")
    name = _entry(header, b"object") or selected.name
    if not field_class:
        raise OpenFOAMDecodeError("field is missing its OpenFOAM class")
    format_name = _entry(header, b"format") or "ascii"
    values = _uniform_value(raw)
    uniform = values is not None
    if values is None:
        values = _nonuniform_value(raw, format_name == "binary")
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
