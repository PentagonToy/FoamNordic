"""Model contracts and exporters.

ONNX is the first native backend. Joblib and Equinox adapters will live in this
namespace so they can share contracts without adding their dependencies to the
core runtime.
"""

from .. import export
from ..export import Tensor

__all__ = ["Tensor", "export"]


def __dir__() -> list[str]:
    return sorted(__all__)
