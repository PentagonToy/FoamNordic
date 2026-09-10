"""Shared public path input normalization."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias


PathInput: TypeAlias = str | PathLike[str]


def path_from(value: PathInput, *, expand_user: bool = True) -> Path:
    """Normalize strings, pathlib paths, and third-party PathLike objects."""

    if isinstance(value, bytes):
        raise TypeError("paths must be text, not bytes")
    path = Path(value)
    return path.expanduser() if expand_user else path
