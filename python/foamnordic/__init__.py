"""Declarative control plane for native FoamNordic workloads."""

__version__ = "1.0.3.dev6"

from . import export, math, models, openfoam, postprocess, random, runtime
from .export import Tensor
from .core.expressions import FieldExpression, field, filter_width, grad
from .core.plan import CompiledPlan
from .execution.observe import FieldSummary, ObservationRecord, ObservationStream, ObservationTiming
from .execution.run import Result, ResultArtifacts, Run, RunStatus, RunSummary
from .core.spec import (
    Attached,
    Closure,
    Longship,
    Observe,
    Operator,
    Slurm,
    Transform,
)
from .math import Math
from .random import Random

OpenFOAM = openfoam
Export = export
Models = models
Runtime = runtime
Postprocess = postprocess

__all__ = [
    "Attached",
    "Closure",
    "CompiledPlan",
    "FieldExpression",
    "FieldSummary",
    "Longship",
    "Math",
    "Models",
    "OpenFOAM",
    "Observe",
    "Operator",
    "ObservationRecord",
    "ObservationStream",
    "ObservationTiming",
    "Result",
    "ResultArtifacts",
    "Run",
    "RunStatus",
    "RunSummary",
    "Runtime",
    "Postprocess",
    "Random",
    "Slurm",
    "Tensor",
    "Transform",
    "Export",
    "__version__",
    "export",
    "field",
    "filter_width",
    "grad",
    "models",
    "openfoam",
    "postprocess",
    "random",
    "runtime",
]


def __dir__() -> list[str]:
    """Keep interactive discovery focused on the supported public API."""

    return sorted(__all__)
