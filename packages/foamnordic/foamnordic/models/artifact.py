"""Read-only public ownership boundary for FoamNordic model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from bindnordic.models.artifact import load as load_artifact


@dataclass(frozen=True, slots=True)
class TensorContract:
    """One named tensor port in an FNOM execution contract."""

    name: str
    components: int
    dtype: str


@dataclass(frozen=True, slots=True)
class FnomArtifact:
    """Validated, read-only metadata for one FoamNordic model artifact."""

    path: Path
    name: str
    format: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    runtime: str | None
    input_scaler: str | None
    output_scaler: str | None
    bundled: bool
    payload_offset: int | None
    payload_size: int | None

    @property
    def backend(self) -> str:
        """Return the embedded model backend family."""

        return self.format

    @property
    def executable_payload(self) -> bool:
        """Return whether loading the payload may execute supplied code."""

        return self.format in {"joblib", "equinox"} or (
            self.format == "compiled" and self.runtime != "forest"
        )

    @property
    def trust_policy(self) -> str:
        """Return the execution policy required by this backend."""

        return "explicit-trust-required" if self.executable_payload else "data-only"

    @property
    def size(self) -> int:
        """Return the complete FNOM file size in bytes."""

        return self.path.stat().st_size

    def validate(self) -> FnomArtifact:
        """Re-read and validate this artifact without executing its payload."""

        return load(self.path)


def load(path: str | Path) -> FnomArtifact:
    """Load and structurally validate one FNOM artifact.

    Loading never imports or executes an embedded Joblib, Equinox, or ONNX
    payload. The native FNOM reader validates the container and manifest.
    """

    artifact = load_artifact(path)

    return FnomArtifact(
        path=artifact.path,
        name=artifact.name,
        format=artifact.format,
        inputs=tuple(TensorContract(value.name, value.components, value.dtype) for value in artifact.inputs),
        outputs=tuple(TensorContract(value.name, value.components, value.dtype) for value in artifact.outputs),
        runtime=artifact.runtime,
        input_scaler=None if artifact.input_scaler is None else artifact.input_scaler.kind,
        output_scaler=None if artifact.output_scaler is None else artifact.output_scaler.kind,
        bundled=True,
        payload_offset=artifact.payload_offset,
        payload_size=artifact.payload_size,
    )


__all__ = ["FnomArtifact", "TensorContract", "load"]
