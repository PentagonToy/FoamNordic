"""Declarative control plane for native FoamNordic workloads."""

# Compatibility names remain importable without cluttering interactive discovery.
# ruff: noqa: F401

__version__ = "1.0.11"

from . import export, math, models, openfoam, postprocess, random, runtime
from .core.expressions import (
    Context,
    Field,
    FieldExpression,
    FieldSelection,
    field,
    fields,
    filter_width,
    grad,
    patch,
)
from .core.plan import CompiledPlan
from .core.spec import (
    Attached,
    Closure,
    Longship,
    Observe,
    Operator,
    Slurm,
    Transform,
)
from .execution.observe import (
    FieldSummary,
    ObservationRecord,
    ObservationStream,
    ObservationTiming,
)
from .execution.run import Result, ResultArtifacts, Run, RunStatus, RunSummary
from .export import Tensor
from .math import Math
from .random import Random

OpenFOAM = openfoam
Export = export
Models = models
Runtime = runtime
Postprocess = postprocess

__all__ = [
    "Closure",
    "Context",
    "Export",
    "Field",
    "Longship",
    "Math",
    "Models",
    "Observe",
    "OpenFOAM",
    "Operator",
    "Postprocess",
    "Random",
    "Runtime",
    "Slurm",
    "Tensor",
    "Transform",
    "__version__",
]


def __dir__() -> list[str]:
    """Keep interactive discovery focused on the supported public API."""

    return sorted(__all__)
