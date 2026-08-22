"""Declarative control plane for native FoamNordic workloads."""

__version__ = "1.0.3.dev4"

from . import export, math, models, openfoam, postprocess, runtime
from .export import Tensor
from ._expressions import FieldExpression, field, filter_width, grad
from ._plan import CompiledPlan
from ._observe import FieldSummary, ObservationRecord, ObservationStream, ObservationTiming
from ._run import Result, ResultArtifacts, Run, RunStatus, RunSummary
from ._spec import Attached, Closure, Longship, Observe, Retention, Slurm
from .math import Math

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
    "ObservationRecord",
    "ObservationStream",
    "ObservationTiming",
    "Retention",
    "Result",
    "ResultArtifacts",
    "Run",
    "RunStatus",
    "RunSummary",
    "Runtime",
    "Postprocess",
    "Slurm",
    "Tensor",
    "Export",
    "__version__",
    "export",
    "field",
    "filter_width",
    "grad",
    "models",
    "openfoam",
    "postprocess",
    "runtime",
]


def __dir__() -> list[str]:
    """Keep interactive discovery focused on the supported public API."""

    return sorted(__all__)
