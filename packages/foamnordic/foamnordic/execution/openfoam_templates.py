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


_SUBDOMAINS = re.compile(r"\bnumberOfSubdomains\s+([0-9]+)\s*;")
_COMMENTS = re.compile(r"/\*[\s\S]*?\*/|//[^\r\n]*")


def _without_comments(text: str) -> str:
    # Preserve offsets so edits can target the original dictionary
    return _COMMENTS.sub(lambda comment: " " * len(comment[0]), text)


def decomposition_subdomains(path: Path) -> int | None:
    """Read a literal ``numberOfSubdomains`` from an OpenFOAM dictionary."""

    if not path.is_file():
        return None
    match = _SUBDOMAINS.search(_without_comments(path.read_text(encoding="utf-8")))
    return int(match[1]) if match is not None else None


def _decomposition_method(text: str) -> str | None:
    match = re.search(
        r"\bmethod\s+([A-Za-z][A-Za-z0-9_]*)\s*;",
        text,
    )
    return match.group(1) if match is not None else None


def _decomposition_factors(text: str, method: str) -> tuple[int, int, int] | None:
    block = re.search(
        rf"\b{re.escape(method)}(?:Coeffs)?\s*\{{(?P<body>.*?)\}}",
        text,
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

    text = path.read_text(encoding="utf-8")
    clean = _without_comments(text)
    method = _decomposition_method(clean)
    if method in {"simple", "hierarchical"}:
        factors = _decomposition_factors(clean, method)
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

    count = _SUBDOMAINS.search(clean)
    if count is not None:
        if int(count[1]) == ranks:
            return False
        start, end = count.span(1)
        path.write_text(f"{text[:start]}{ranks}{text[end:]}", encoding="utf-8")
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
