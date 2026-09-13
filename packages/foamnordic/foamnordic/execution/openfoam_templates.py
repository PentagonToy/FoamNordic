"""OpenFOAM runtime-template access and dictionary preparation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .templates import read as read_template
from .templates import render as render_template


def runtime_template(name: str, *, kind: str) -> str:
    """Read one template consumed while preparing an OpenFOAM run."""

    return read_template(f"openfoam/runtime/{name}", kind=kind)


def decomposition_subdomains(path: Path) -> int | None:
    """Read a literal ``numberOfSubdomains`` from an OpenFOAM dictionary."""

    if not path.is_file():
        return None
    text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    match = re.search(r"\bnumberOfSubdomains\s+([0-9]+)\s*;", text)
    return int(match.group(1)) if match is not None else None


def prepare_decomposition(path: Path, ranks: int) -> bool:
    """Keep a compatible policy or install a portable run-local policy."""

    if decomposition_subdomains(path) == ranks:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_template(
            runtime_template(
                "decomposeParDict.in",
                kind="OpenFOAM decomposition",
            ),
            {
                "NUMBER_OF_SUBDOMAINS": ranks,
                "DECOMPOSITION_METHOD": "scotch",
                "METHOD_COEFFICIENTS": "",
            },
            kind="OpenFOAM decomposition",
        ),
        encoding="utf-8",
    )
    return True


def derived_scheme_defaults() -> dict[str, dict[str, str]]:
    """Read conservative schemes for derived OpenFOAM expressions."""

    value = json.loads(
        runtime_template(
            "derivedSchemes.json",
            kind="OpenFOAM derived-scheme",
        )
    )
    if not isinstance(value, dict):
        raise TypeError("derived-scheme template must contain a mapping")
    return value


__all__ = [
    "decomposition_subdomains",
    "derived_scheme_defaults",
    "prepare_decomposition",
    "runtime_template",
]
