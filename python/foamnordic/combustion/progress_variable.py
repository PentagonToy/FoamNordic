"""Progress-variable combustion orchestration declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from ..core.expressions import FieldExpression, FieldSelection, field as bind_field
from ..core.spec import Closure, Operator
from ..core.validation import require_nonempty
from .contracts import (
    REACTION_RATE_CONTRACT_VERSION,
    CouplingPolicy,
    validate_reaction_rate,
)
from .manifold import BetaFDF

if TYPE_CHECKING:
    from ..openfoam import Case


@dataclass(frozen=True, slots=True)
class ProgressVariable:
    """Couple a reaction-rate closure to a thermochemical manifold.

    This immutable object defines scientific ownership and call ordering. The
    OpenFOAM combustion adapter remains responsible for equation assembly,
    boundary correction, and ``thermo.correct()`` at the declared native site.
    """

    reaction_rate: Closure
    manifold: BetaFDF
    name: str = "progressVariable"
    coupling: CouplingPolicy = field(default_factory=CouplingPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, "combustion name"))
        validate_reaction_rate(self.reaction_rate)
        if not isinstance(self.manifold, BetaFDF):
            raise TypeError("manifold must be created by Manifold.beta_fdf()")
        if not isinstance(self.coupling, CouplingPolicy):
            raise TypeError("coupling must be a CouplingPolicy")
        reaction_progress = self.reaction_rate.inputs["progress"]
        reaction_variance = self.reaction_rate.inputs["variance"]
        if reaction_progress != self.manifold.progress:
            raise ValueError(
                "reaction-rate and manifold progress bindings must match"
            )
        if reaction_variance != self.manifold.variance:
            raise ValueError(
                "reaction-rate and manifold variance bindings must match"
            )

    def to_plan(self) -> dict[str, object]:
        return {
            "contract": "progress_variable_combustion",
            "contract_version": REACTION_RATE_CONTRACT_VERSION,
            "name": self.name,
            "reaction_rate": self.reaction_rate.to_plan(),
            "manifold": self.manifold.to_plan(),
            "coupling": self.coupling.to_plan(),
        }

    def programs(self, case: Case) -> tuple[Closure, Closure]:
        """Lower the scientific contract to two resident field programs.

        Field-family outputs are expanded deterministically against the
        initial OpenFOAM object registry before any process is launched.
        """

        manifold_inputs: dict[str, FieldExpression] = {
            "progress": self.manifold.progress,
            "variance": self.manifold.variance,
            **self.manifold.conditioning,
        }
        manifold_outputs: dict[str, FieldExpression] = {}
        for logical_name, binding in self.manifold.outputs.items():
            if isinstance(binding, FieldExpression):
                assert binding.field_name is not None
                metadata = case.field(binding.field_name)
                if metadata.field_class != "volScalarField":
                    raise ValueError(
                        f"manifold output {binding.field_name!r} must be a "
                        f"volScalarField, got {metadata.field_class}"
                    )
                manifold_outputs[logical_name] = binding
                continue
            assert isinstance(binding, FieldSelection)
            matches = tuple(
                name for name in case.fields if fnmatchcase(name, binding.pattern)
            )
            if not matches:
                raise ValueError(
                    f"manifold field selection {binding.pattern!r} matched no "
                    "initial OpenFOAM fields"
                )
            for name in matches:
                metadata = case.field(name)
                if metadata.field_class != "volScalarField":
                    raise ValueError(
                        f"manifold output {name!r} must be a volScalarField, "
                        f"got {metadata.field_class}"
                    )
                if name in manifold_outputs:
                    raise ValueError(
                        f"manifold output expansion repeats logical port {name!r}"
                    )
                manifold_outputs[name] = bind_field(name)

        manifold_program = Closure(
            name=f"{self.name}Manifold",
            operator=Operator.model(self.manifold.table),
            inputs=manifold_inputs,
            outputs=manifold_outputs,
        )
        return self.reaction_rate, manifold_program
