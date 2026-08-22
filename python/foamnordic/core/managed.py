"""Ownership markers for safely removable FoamNordic output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


MARKER = ".foamnordic-generated.json"


def mark_generated(
    path: Path,
    *,
    kind: str,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Mark an exact directory as generated and therefore clobberable."""

    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "root": str(root),
    }
    if metadata:
        value["metadata"] = dict(metadata)
    marker.write_text(
        json.dumps(value, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return marker


def generated_kind(path: Path) -> str | None:
    """Return a valid marker kind, rejecting copied or stale markers."""

    root = Path(path).expanduser().resolve()
    marker = root / MARKER
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if value.get("schema_version") != 1 or value.get("root") != str(root):
        return None
    kind = value.get("kind")
    return kind if isinstance(kind, str) and kind else None
