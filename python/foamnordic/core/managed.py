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


def relocate_generated(path: Path, *, previous: Path) -> None:
    """Update a valid ownership marker after an atomic directory rename."""

    root = Path(path).expanduser().resolve()
    old_root = Path(previous).expanduser().resolve()
    marker = root / MARKER
    if not marker.exists():
        # Low-level lifecycle tests and internal callers may launch from an
        # unmanaged scratch directory. There is no ownership claim to update.
        return
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"FoamNordic ownership marker is unreadable: {marker}"
        ) from error
    if value.get("schema_version") != 1 or value.get("root") != str(old_root):
        raise RuntimeError(
            "FoamNordic refused to relocate an invalid ownership marker: "
            f"{marker}"
        )
    value["root"] = str(root)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
