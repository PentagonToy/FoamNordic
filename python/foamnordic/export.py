"""Native model artifact export with quiet-by-default reporting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from .core.paths import PathInput, path_from
from .core.validation import require_nonempty, require_positive

try:
    from . import _native
except ImportError:
    _native = None


__all__ = ["Tensor", "equinox", "joblib", "onnx"]


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


def _write_onnx_payload(model: object, destination: Path) -> int:
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
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    verbose: bool = False,
) -> Path:
    """Write an ONNX payload plus its uncompressed native `.fnom` manifest."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    manifest, model_path, default_name = _destination(path, ".onnx")
    model_name = require_nonempty(name or default_name, "model name")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{model_path.name}.", dir=model_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        payload_size = _write_onnx_payload(model, temporary)
        if payload_size == 0:
            raise ValueError("ONNX payload must not be empty")
        temporary.replace(model_path)
        dtype = _manifest_contract(
            manifest,
            model_path,
            model_name,
            "onnx",
            inputs,
            outputs,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
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


def _manifest_contract(
    manifest: Path,
    payload: Path,
    model_name: str,
    model_format: str,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    tree_leaves: Sequence[tuple[str, str, list[int], int, int]] = (),
    *,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
) -> str:
    if _native is None:
        raise RuntimeError("native artifact export requires a FoamNordic binary wheel")
    input_contract = _contracts(inputs, "inputs")
    output_contract = _contracts(outputs, "outputs")
    dtypes = {tensor.dtype for tensor in (*inputs.values(), *outputs.values())}
    if len(dtypes) != 1:
        raise ValueError("all tensors in one native artifact must share a dtype")
    dtype = dtypes.pop()
    _native.write_model_manifest(
        str(manifest),
        payload.name,
        model_name,
        model_format,
        input_contract,
        output_contract,
        dtype,
        list(tree_leaves),
        x_scaler,
        y_scaler,
    )
    return dtype


def _destination(path: PathInput, suffix: str) -> tuple[Path, Path, str]:
    manifest = path_from(path).resolve()
    if manifest.suffix.lower() != ".fnom":
        raise ValueError("FoamNordic native manifests must use the .fnom suffix")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    return manifest, manifest.with_suffix(suffix), manifest.stem


def _temporary_for(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _copy_path_payload(model: object, destination: Path, label: str) -> bool:
    if not isinstance(model, (str, os.PathLike)):
        return False
    source = path_from(model).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} payload does not exist: {source}")
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
    return True


def joblib(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    verbose: bool = False,
) -> Path:
    """Export one path-backed or fitted Joblib model without archive copies."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    manifest, payload, default_name = _destination(path, ".joblib")
    model_name = require_nonempty(name or default_name, "model name")
    temporary = _temporary_for(payload)
    try:
        if not _copy_path_payload(model, temporary, "Joblib"):
            try:
                import joblib as joblib_module
            except ImportError as error:
                raise ImportError("Joblib is missing from this FoamNordic installation") from error
            if not callable(getattr(model, "predict", None)) and not callable(model):
                raise TypeError("Joblib models must define predict() or be callable")
            joblib_module.dump(model, temporary, compress=0, protocol=5)
        if temporary.stat().st_size == 0:
            raise ValueError("Joblib payload must not be empty")
        temporary.replace(payload)
        dtype = _manifest_contract(
            manifest,
            payload,
            model_name,
            "joblib",
            inputs,
            outputs,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest, payload, model_name, inputs, outputs, dtype, format_name="Joblib"
        )
    return manifest


def _equinox_leaves(model: object) -> list[tuple[str, str, list[int], int, int]]:
    try:
        import jax
    except ImportError as error:
        raise ImportError("JAX is missing from this FoamNordic installation") from error
    result = []
    offset = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        shape = getattr(leaf, "shape", None)
        dtype_value = getattr(leaf, "dtype", None)
        if shape is None or dtype_value is None or not shape:
            continue
        dtype = str(dtype_value)
        if dtype not in {"float32", "float64", "int32", "int64"}:
            continue
        count = int(getattr(leaf, "size")) * int(getattr(dtype_value, "itemsize"))
        if count == 0:
            continue
        result.append((jax.tree_util.keystr(path), dtype, list(shape), offset, count))
        offset += count
    if not result:
        raise ValueError("Equinox model must contain at least one supported array leaf")
    return result


def equinox(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    batched: bool = False,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    verbose: bool = False,
) -> Path:
    """Export a reconstructable Equinox PyTree for a resident JAX worker."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    if not isinstance(batched, bool):
        raise TypeError("batched must be a boolean")
    if not callable(model):
        raise TypeError("Equinox models must be callable")
    try:
        import cloudpickle
        import equinox as _  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Equinox is missing from this FoamNordic installation"
        ) from error
    manifest, payload, default_name = _destination(path, ".eqx")
    model_name = require_nonempty(name or default_name, "model name")
    leaves = _equinox_leaves(model)
    temporary = _temporary_for(payload)
    try:
        with temporary.open("wb") as stream:
            cloudpickle.dump(
                {
                    "schema": "foamnordic.equinox/v1",
                    "model": model,
                    "batched": batched,
                },
                stream,
                protocol=5,
            )
        if temporary.stat().st_size == 0:
            raise ValueError("Equinox payload must not be empty")
        temporary.replace(payload)
        dtype = _manifest_contract(
            manifest,
            payload,
            model_name,
            "equinox",
            inputs,
            outputs,
            leaves,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest, payload, model_name, inputs, outputs, dtype, format_name="Equinox"
        )
    return manifest


def _display_export(
    manifest: Path,
    model: Path,
    name: str,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    dtype: str,
    *,
    format_name: str = "ONNX",
) -> None:
    import onsaemiro as osm

    metadata = _native.read_model_manifest(str(manifest))
    input_scaler = metadata["input_scaler"]
    output_scaler = metadata["output_scaler"]

    table = osm.TableMaker(
        title=f"FoamNordic Model Exported: {name}",
        columns=["Property", "Value"],
        mode="static",
    )
    rows = (
        ("Manifest", manifest.name),
        ("Payload", model.name),
        ("Format", f"{format_name} + FNOM v1"),
        ("Dtype", dtype),
        ("Inputs", ", ".join(f"{key}[{value.components}]" for key, value in inputs.items())),
        ("Outputs", ", ".join(f"{key}[{value.components}]" for key, value in outputs.items())),
        ("Input scaler", "none" if input_scaler is None else input_scaler["kind"]),
        ("Output scaler", "none" if output_scaler is None else output_scaler["kind"]),
        ("Compression", "none (path-backed startup)"),
        ("Payload size", f"{model.stat().st_size} B"),
    )
    for row in rows:
        table.add_row(row)
    table.display()


def __dir__() -> list[str]:
    return sorted(__all__)
