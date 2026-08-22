"""Stable serialized representation of a compiled Longship declaration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ._paths import PathInput, path_from


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """An immutable, content-addressed run plan."""

    _canonical_json: str
    digest: str

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "CompiledPlan":
        canonical = _canonical_bytes(value)
        digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        return cls(canonical.decode("utf-8"), digest)

    @property
    def schema_version(self) -> int:
        return int(self.as_dict()["schema_version"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    def to_json(self, *, indent: int | None = 2) -> str:
        if indent is None:
            return self._canonical_json
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    def write(self, path: PathInput) -> Path:
        destination = path_from(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination
