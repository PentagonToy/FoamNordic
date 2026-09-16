"""Durable OpenFOAM field access, dataset conversion, and case comparison."""

from .case import Case
from .comparison import compare
from . import dataset

__all__ = ["Case", "compare", "dataset"]


def __dir__() -> list[str]:
    return sorted(__all__)
