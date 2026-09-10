"""Model artifacts, payloads and compilation without a solver dependency."""

from .artifact import (
    AffineScaler,
    ModelArtifact,
    TensorContract,
    TreeLeaf,
    extract_payload,
    load,
    payload_region,
    read_payload,
    register_scaler_adapter,
    write_bundle,
)
from .compiled import generate_source, supports
from .payloads import write_equinox, write_joblib, write_onnx
from .schema import MODEL_FORMATS, ModelFormatSpec, model_format
from .runtime import (
    EmbeddedPayload,
    compiled_evaluator,
    configure_model_threads,
    equinox_evaluator,
    joblib_evaluator,
    packed_result,
)

__all__ = [
    "AffineScaler",
    "ModelArtifact",
    "MODEL_FORMATS",
    "ModelFormatSpec",
    "TensorContract",
    "TreeLeaf",
    "EmbeddedPayload",
    "compiled_evaluator",
    "configure_model_threads",
    "equinox_evaluator",
    "extract_payload",
    "generate_source",
    "load",
    "model_format",
    "joblib_evaluator",
    "packed_result",
    "payload_region",
    "read_payload",
    "register_scaler_adapter",
    "supports",
    "write_bundle",
    "write_equinox",
    "write_joblib",
    "write_onnx",
]
