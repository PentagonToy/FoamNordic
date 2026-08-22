"""Progress-variable combustion orchestration declarations."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.spec import Closure
from ..core.validation import require_nonempty
from .contracts import (
    REACTION_RATE_CONTRACT_VERSION,
    CouplingPolicy,
    validate_reaction_rate,
)
from .manifold import BetaFDF


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
