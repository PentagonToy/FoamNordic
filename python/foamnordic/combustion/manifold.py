"""Presumed-FDF and manifold declarations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..core.expressions import FieldExpression
from ..core.paths import PathInput, path_from
from .contracts import (
    MANIFOLD_CONTRACT_VERSION,
    ManifoldOutput,
    freeze_manifold_outputs,
)


@dataclass(frozen=True, slots=True)
class BetaFDF:
    """A beta-FDF composition closure backed by one FNOM table artifact.

    The initial contract deliberately describes a pre-integrated table. It
    does not perform per-cell Python quadrature or claim to update OpenFOAM
    thermodynamics by itself.
    """

    table: PathInput
    progress: FieldExpression
    variance: FieldExpression
    outputs: Mapping[str, ManifoldOutput]
    conditioning: Mapping[str, FieldExpression]
    bounds: str = "clip"
    integration: str = "preintegrated"

    def __post_init__(self) -> None:
        table = path_from(self.table)
        if table.suffix.lower() != ".fnom":
            raise ValueError("beta_fdf table must be an .fnom manifest")
        object.__setattr__(self, "table", table)
        for binding, label in (
            (self.progress, "progress"),
            (self.variance, "variance"),
        ):
            if not isinstance(binding, FieldExpression):
                raise TypeError(f"beta_fdf {label} must be a FieldExpression")
        object.__setattr__(self, "outputs", freeze_manifold_outputs(self.outputs))
        normalized: dict[str, FieldExpression] = {}
        for name, binding in self.conditioning.items():
            logical_name = name.strip()
            if not logical_name:
                raise ValueError("conditioning name must not be empty")
            if not isinstance(binding, FieldExpression):
                raise TypeError(
                    f"conditioning input {logical_name!r} must be a FieldExpression"
                )
            normalized[logical_name] = binding
        object.__setattr__(self, "conditioning", MappingProxyType(normalized))
        if self.bounds not in {"clip", "error"}:
            raise ValueError("beta_fdf bounds must be clip or error")
        if self.integration != "preintegrated":
            raise ValueError(
                "the initial beta_fdf contract supports preintegrated tables only"
            )

    def to_plan(self) -> dict[str, object]:
        return {
            "contract_version": MANIFOLD_CONTRACT_VERSION,
            "kind": "beta_fdf",
            "table": str(self.table),
            "inputs": {
                "progress": self.progress.to_plan(),
                "variance": self.variance.to_plan(),
                "conditioning": {
                    name: binding.to_plan()
                    for name, binding in self.conditioning.items()
                },
            },
            "outputs": {
                name: binding.to_plan() for name, binding in self.outputs.items()
            },
            "bounds": self.bounds,
            "integration": self.integration,
        }


class Manifold:
    """Factories for thermochemical manifold declarations."""

    @staticmethod
    def beta_fdf(
        *,
        table: PathInput,
        progress: FieldExpression,
        variance: FieldExpression,
        outputs: Mapping[str, ManifoldOutput],
        conditioning: Mapping[str, FieldExpression] | None = None,
        bounds: str = "clip",
    ) -> BetaFDF:
        """Declare a pre-integrated beta-FDF table closure."""

        return BetaFDF(
            table=table,
            progress=progress,
            variance=variance,
            outputs=outputs,
            conditioning={} if conditioning is None else conditioning,
            bounds=bounds,
        )

    def __new__(cls, *_args, **_kwargs):
        raise TypeError("Manifold is a factory namespace; use Manifold.beta_fdf()")

    @classmethod
    def __dir__(cls) -> list[str]:
        return ["beta_fdf"]
