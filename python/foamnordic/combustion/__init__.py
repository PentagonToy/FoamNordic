"""Declarative progress-variable combustion contracts."""

from .contracts import CouplingPolicy
from .manifold import BetaFDF, Manifold
from .progress_variable import ProgressVariable

__all__ = ["BetaFDF", "CouplingPolicy", "Manifold", "ProgressVariable"]


def __dir__() -> list[str]:
    return sorted(__all__)
