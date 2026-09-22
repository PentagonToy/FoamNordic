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


def _decomposition_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*$", "", text, flags=re.MULTILINE)


def _decomposition_method(text: str) -> str | None:
    match = re.search(
        r"\bmethod\s+([A-Za-z][A-Za-z0-9_]*)\s*;",
        _without_comments(text),
    )
    return match.group(1) if match is not None else None


def _decomposition_factors(text: str, method: str) -> tuple[int, int, int] | None:
    clean = _without_comments(text)
    block = re.search(
        rf"\b{re.escape(method)}Coeffs\s*\{{(?P<body>.*?)\}}",
        clean,
        flags=re.DOTALL,
    )
    if block is None:
        return None
    factors = re.search(
        r"\bn\s*\(\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*\)\s*;",
        block.group("body"),
    )
    if factors is None:
        return None
    return (
        int(factors.group(1)),
        int(factors.group(2)),
        int(factors.group(3)),
    )


def _portable_decomposition(path: Path, ranks: int) -> None:
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


def prepare_decomposition(path: Path, ranks: int) -> bool:
    """Prepare a rank-compatible policy inside the isolated run case."""

    if not path.is_file():
        _portable_decomposition(path, ranks)
        return True

    text = _decomposition_text(path)
    method = _decomposition_method(text)
    if method in {"simple", "hierarchical"}:
        factors = _decomposition_factors(text, method)
        factor_count = (
            None if factors is None else factors[0] * factors[1] * factors[2]
        )
        if factor_count != ranks:
            declared = "missing" if factors is None else " x ".join(map(str, factors))
            print(
                f"[FoamNordic] {method} decomposition n=({declared}) does not "
                f"match ranks={ranks}; using scotch for the isolated run."
            )
            _portable_decomposition(path, ranks)
            return True

    current = decomposition_subdomains(path)
    if current == ranks:
        return False

    if current is not None:
        updated = re.sub(
            r"(\bnumberOfSubdomains\s+)[0-9]+(\s*;)",
            rf"\g<1>{ranks}\g<2>",
            text,
            count=1,
        )
        path.write_text(updated, encoding="utf-8")
        return True

    print(
        "[FoamNordic] decomposeParDict has no literal numberOfSubdomains; "
        "using scotch for the isolated run."
    )
    _portable_decomposition(path, ranks)
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
