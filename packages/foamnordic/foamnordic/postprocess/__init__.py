"""Durable OpenFOAM field access, statistics, and case comparison."""

from .case import Case
from .comparison import compare

__all__ = ["Case", "compare"]


def __dir__() -> list[str]:
    return sorted(__all__)
