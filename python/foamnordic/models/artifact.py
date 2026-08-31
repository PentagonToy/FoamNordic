"""Read-only public ownership boundary for FoamNordic model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    from .. import _native
except ImportError:
    _native = None


_BUNDLE_MAGIC = {
    b"FNOBND1\0": 1,
    b"FNOBND2\0": 2,
}


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
    container_version: int | None
    schema_version: int
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
    def size(self) -> int:
        """Return the complete FNOM file size in bytes."""

        return self.path.stat().st_size

    def validate(self) -> FnomArtifact:
        """Re-read and validate this artifact without executing its payload."""

        return load(self.path)


def _native_module():
    if _native is None:
        raise RuntimeError(
            "FNOM inspection requires FoamNordic's native extension; install "
            "a binary wheel or rebuild the package"
        )
    return _native


def _scaler_kind(value: object) -> str | None:
    if value is None:
        return None
    return str(dict(value)["kind"])


def _ports(values: object) -> tuple[TensorContract, ...]:
    return tuple(
        TensorContract(str(name), int(components), str(dtype))
        for name, components, dtype in values
    )


def load(path: str | Path) -> FnomArtifact:
    """Load and structurally validate one FNOM artifact.

    Loading never imports or executes an embedded Joblib, Equinox, or ONNX
    payload. The native FNOM reader validates the container and manifest.
    """

    selected = Path(path).expanduser().resolve()
    if selected.suffix.lower() != ".fnom":
        raise ValueError("FoamNordic model artifacts must use the .fnom extension")
    if not selected.is_file():
        raise FileNotFoundError(f"FNOM artifact does not exist: {selected}")

    native = _native_module()
    metadata: Mapping[str, object] = native.read_model_manifest(str(selected))
    with selected.open("rb") as stream:
        prefix = stream.read(8)
    container_version = _BUNDLE_MAGIC.get(prefix)
    bundled = bool(metadata["bundled"])
    payload_offset: int | None = None
    payload_size: int | None = None
    if bundled:
        payload_offset, payload_size = (
            int(value) for value in native.read_model_payload_region(str(selected))
        )

    return FnomArtifact(
        path=selected,
        container_version=container_version,
        schema_version=int(metadata["schema_version"]),
        name=str(metadata["name"]),
        format=str(metadata["format"]),
        inputs=_ports(metadata["inputs"]),
        outputs=_ports(metadata["outputs"]),
        runtime=None if metadata["runtime"] is None else str(metadata["runtime"]),
        input_scaler=_scaler_kind(metadata["input_scaler"]),
        output_scaler=_scaler_kind(metadata["output_scaler"]),
        bundled=bundled,
        payload_offset=payload_offset,
        payload_size=payload_size,
    )


__all__ = ["FnomArtifact", "TensorContract", "load"]
