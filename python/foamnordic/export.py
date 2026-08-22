"""Native model artifact export with quiet-by-default reporting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ._paths import PathInput, path_from
from ._validation import require_nonempty, require_positive

try:
    from . import _native
except ImportError:
    _native = None


@dataclass(frozen=True, slots=True)
class Tensor:
    """One packed model tensor contract."""

    components: int = 1
    dtype: str = "float64"

    def __post_init__(self) -> None:
        require_positive(self.components, "tensor components")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("tensor dtype must be float32 or float64")

    @classmethod
    def scalar(cls, *, dtype: str = "float64") -> Tensor:
        return cls(1, dtype)

    @classmethod
    def vector(cls, *, components: int = 3, dtype: str = "float64") -> Tensor:
        return cls(components, dtype)

    @classmethod
    def tensor(cls, *, components: int = 9, dtype: str = "float64") -> Tensor:
        return cls(components, dtype)


def _contracts(value: Mapping[str, Tensor], label: str) -> list[tuple[str, int]]:
    if not value:
        raise ValueError(f"{label} must not be empty")
    result = []
    for name, tensor in value.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{label}[{name!r}] must be a Tensor")
        result.append((require_nonempty(name, f"{label} name"), tensor.components))
    return result


def _write_payload(model: object, destination: Path) -> int:
    """Write without loading path-backed, potentially multi-GB models into RAM."""

    if isinstance(model, (str, os.PathLike)):
        source = path_from(model).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ONNX payload does not exist: {source}")
        with source.open("rb") as input_stream, destination.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        return destination.stat().st_size
    if isinstance(model, bytes):
        destination.write_bytes(model)
        return len(model)
    serialize = getattr(model, "SerializeToString", None)
    if callable(serialize):
        value = serialize()
        if isinstance(value, bytes):
            destination.write_bytes(value)
            return len(value)
    raise TypeError(
        "model must be ONNX bytes, an ONNX path, or expose SerializeToString(); "
        "callable-to-ONNX lowering remains a separate exporter backend"
    )


def onnx(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    verbose: bool = False,
) -> Path:
    """Write an ONNX payload plus its uncompressed native `.fnom` manifest."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    if _native is None:
        raise RuntimeError("native artifact export requires a FoamNordic binary wheel")
    manifest = path_from(path).resolve()
    if manifest.suffix.lower() != ".fnom":
        raise ValueError("FoamNordic native manifests must use the .fnom suffix")
    input_contract = _contracts(inputs, "inputs")
    output_contract = _contracts(outputs, "outputs")
    dtypes = {tensor.dtype for tensor in (*inputs.values(), *outputs.values())}
    if len(dtypes) != 1:
        raise ValueError("all tensors in one native artifact must share a dtype")
    dtype = dtypes.pop()
    model_name = require_nonempty(name or manifest.stem, "model name")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    model_path = manifest.with_suffix(".onnx")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{model_path.name}.", dir=model_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        payload_size = _write_payload(model, temporary)
        if payload_size == 0:
            raise ValueError("ONNX payload must not be empty")
        temporary.replace(model_path)
        _native.write_onnx_manifest(
            str(manifest),
            model_path.name,
            model_name,
            input_contract,
            output_contract,
            dtype,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest,
            model_path,
            model_name,
            inputs,
            outputs,
            dtype,
        )
    return manifest


def _display_export(
    manifest: Path,
    model: Path,
    name: str,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    dtype: str,
) -> None:
    import onsaemiro as osm

    table = osm.TableMaker(
        title=f"FoamNordic Model Exported: {name}",
        columns=["Property", "Value"],
        mode="static",
    )
    rows = (
        ("Manifest", manifest.name),
        ("Payload", model.name),
        ("Format", "ONNX + FNOM v1"),
        ("Dtype", dtype),
        ("Inputs", ", ".join(f"{key}[{value.components}]" for key, value in inputs.items())),
        ("Outputs", ", ".join(f"{key}[{value.components}]" for key, value in outputs.items())),
        ("Compression", "none (native startup path)"),
        ("Payload size", f"{model.stat().st_size} B"),
    )
    for row in rows:
        table.add_row(row)
    table.display()
