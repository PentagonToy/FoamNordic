"""Small, dependency-free rendering for FoamNordic generation templates."""

from __future__ import annotations

from collections.abc import Mapping
import re


_TOKEN = re.compile(r"@[A-Z][A-Z0-9_]*@")


def render(template: str, variables: Mapping[str, object], *, kind: str) -> str:
    """Replace ``@UPPER_SNAKE_CASE@`` tokens and reject incomplete output."""

    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace(f"@{name}@", str(value))
    unresolved = sorted(set(_TOKEN.findall(rendered)))
    if unresolved:
        raise ValueError(f"unresolved {kind} template variables: {unresolved}")
    return rendered
