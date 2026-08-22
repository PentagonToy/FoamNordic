"""Binding-neutral contracts shared by combustion declarations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..core.expressions import FieldExpression, FieldSelection
from ..core.spec import Closure


REACTION_RATE_CONTRACT_VERSION = 1
MANIFOLD_CONTRACT_VERSION = 1


def validate_reaction_rate(closure: Closure) -> None:
    """Validate semantic ports without prescribing OpenFOAM field names."""

    if not isinstance(closure, Closure):
        raise TypeError("reaction_rate must be a Closure")
    required_inputs = {"progress", "variance", "temperature"}
    missing_inputs = required_inputs - closure.inputs.keys()
    if missing_inputs:
        raise ValueError(
            "reaction-rate closure is missing semantic input port(s): "
            + ", ".join(sorted(missing_inputs))
        )
    if "reaction_rate" not in closure.outputs:
        raise ValueError(
            "reaction-rate closure requires a 'reaction_rate' output port"
        )


ManifoldOutput = FieldExpression | FieldSelection


def freeze_manifold_outputs(
    outputs: Mapping[str, ManifoldOutput],
) -> Mapping[str, ManifoldOutput]:
    if not outputs:
        raise ValueError("manifold outputs must not be empty")
    normalized: dict[str, ManifoldOutput] = {}
    for name, binding in outputs.items():
        logical_name = name.strip()
        if not logical_name:
            raise ValueError("manifold output name must not be empty")
        if not isinstance(binding, (FieldExpression, FieldSelection)):
            raise TypeError(
                f"manifold output {logical_name!r} must be a field or field selection"
            )
        if isinstance(binding, FieldExpression) and binding.operation != "field":
            raise ValueError(
                f"manifold output {logical_name!r} must bind to a mutable field"
            )
        normalized[logical_name] = binding
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class CouplingPolicy:
    """Numerical ordering contract for a progress-variable combustion loop."""

    correction_stage: str = "outer_corrector"
    source_treatment: str = "lagged"
    thermo_correction: bool = True

    def __post_init__(self) -> None:
        if self.correction_stage != "outer_corrector":
            raise ValueError(
                "the combustion scaffold currently supports outer_corrector only"
            )
        if self.source_treatment != "lagged":
            raise ValueError(
                "the combustion scaffold currently supports lagged sources only"
            )
        if not isinstance(self.thermo_correction, bool):
            raise TypeError("thermo_correction must be a boolean")

    def to_plan(self) -> dict[str, object]:
        return {
            "correction_stage": self.correction_stage,
            "source_treatment": self.source_treatment,
            "thermo_correction": self.thermo_correction,
        }
