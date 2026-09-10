"""Stable FNOM wire-schema descriptors.

Adding a model format is deliberately a schema change: its numeric code must
also be implemented by the native manifest reader. Keeping the rules in one
table prevents exporters and inspectors from growing separate conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelFormatSpec:
    name: str
    code: int
    runtimes: frozenset[str] = frozenset({""})
    requires_tree: bool = False
    executable: bool = False
    data_runtimes: frozenset[str] = frozenset()

    def validate(self, runtime: str | None, has_tree: bool) -> None:
        if (runtime or "") not in self.runtimes:
            raise ValueError(f"{self.name} artifact runtime is invalid")
        if has_tree != self.requires_tree:
            requirement = "require" if self.requires_tree else "forbid"
            raise ValueError(f"{self.name} artifacts {requirement} tree leaves")

    def is_executable(self, runtime: str | None) -> bool:
        return self.executable and (runtime or "") not in self.data_runtimes


_FORMAT_SPECS = (
    ModelFormatSpec("equinox", 1, requires_tree=True, executable=True),
    ModelFormatSpec(
        "joblib",
        2,
        frozenset({"", "sklearn", "sklearnex"}),
        executable=True,
    ),
    ModelFormatSpec("onnx", 3),
    ModelFormatSpec(
        "compiled",
        4,
        frozenset({"compiled", "forest"}),
        executable=True,
        data_runtimes=frozenset({"forest"}),
    ),
)
MODEL_FORMATS = {spec.name: spec for spec in _FORMAT_SPECS}
MODEL_FORMAT_CODES = {spec.code: spec for spec in _FORMAT_SPECS}


def model_format(name: str) -> ModelFormatSpec:
    try:
        return MODEL_FORMATS[name]
    except KeyError as error:
        raise ValueError(f"unsupported model format: {name}") from error


__all__ = ["MODEL_FORMATS", "ModelFormatSpec", "model_format"]
