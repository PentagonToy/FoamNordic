"""Declarative control plane for native FoamNordic workloads."""

__version__ = "1.0.3.dev3"

from . import export, models, openfoam, runtime
from .export import Tensor
from ._expressions import FieldExpression, field, filter_width, grad
from ._plan import CompiledPlan
from ._observe import FieldSummary, ObservationRecord, ObservationStream, ObservationTiming
from ._run import Result, Run, RunStatus, RunSummary
from ._spec import Attached, Closure, Longship, Observe, Retention, Slurm

OpenFOAM = openfoam
Export = export
Models = models
Runtime = runtime

__all__ = [
    "Attached",
    "Closure",
    "CompiledPlan",
    "FieldExpression",
    "FieldSummary",
    "Longship",
    "Models",
    "OpenFOAM",
    "Observe",
    "ObservationRecord",
    "ObservationStream",
    "ObservationTiming",
    "Retention",
    "Result",
    "Run",
    "RunStatus",
    "RunSummary",
    "Runtime",
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
    "runtime",
]


def __dir__() -> list[str]:
    """Keep interactive discovery focused on the supported public API."""

    return sorted(__all__)
