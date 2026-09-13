"""Small, dependency-free rendering for FoamNordic generation templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path, PurePosixPath

_TOKEN = re.compile(r"@[A-Z][A-Z0-9_]*@")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def read(path: str, *, kind: str, default: str | None = None) -> str:
    """Read one packaged template with a source-checkout fallback."""

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("template path must stay inside the template directory")
    packaged = files("foamnordic").joinpath("templates", *relative.parts)
    source = _REPOSITORY_ROOT.joinpath("tools/templates", *relative.parts)
    in_source_checkout = Path(__file__).resolve().is_relative_to(_REPOSITORY_ROOT)
    if in_source_checkout and source.is_file():
        return source.read_text(encoding="utf-8")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    if source.is_file():
        return source.read_text(encoding="utf-8")
    if default is not None:
        return default
    raise RuntimeError(f"FoamNordic {kind} template is unavailable: {path}")


def render(template: str, variables: Mapping[str, object], *, kind: str) -> str:
    """Replace ``@UPPER_SNAKE_CASE@`` tokens and reject incomplete output."""

    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace(f"@{name}@", str(value))
    unresolved = sorted(set(_TOKEN.findall(rendered)))
    if unresolved:
        raise ValueError(f"unresolved {kind} template variables: {unresolved}")
    return rendered
