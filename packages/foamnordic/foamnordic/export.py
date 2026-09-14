"""Native model artifact export with quiet-by-default reporting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from bindnordic.models.artifact import load as load_artifact
from bindnordic.models.artifact import _write_bundle
from bindnordic.models.payloads import write_equinox, write_joblib, write_onnx
from bindnordic.models.table import write_table

from .core.paths import PathInput, path_from
from .core.validation import require_nonempty, require_positive

try:
    from . import _native
except ImportError:
    _native = None


__all__ = ["Tensor", "equinox", "onnx", "sklearn", "table"]


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
    return write_onnx(model, destination)


def table(
    axes: Mapping[str, object],
    values: object,
    *,
    path: PathInput,
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    dtype: str = "float64",
    bounds: str | Mapping[str, str] = "clamp",
    verbose: bool = False,
) -> Path:
    """Write one regular-grid lookup table as a self-contained `.fnom` bundle."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    if not axes:
        raise ValueError("table axes must not be empty")
    inputs = {axis: Tensor.scalar(dtype=dtype) for axis in axes}
    output_features = sum(tensor.components for tensor in outputs.values())
    manifest, payload, default_name = _destination(path, ".table")
    model_name = require_nonempty(name or default_name, "model name")
    try:
        write_table(
            payload,
            axes,
            values,
            output_features,
            dtype=dtype,
            bounds=bounds,
        )
        written_dtype = _manifest_contract(
            manifest,
            payload,
            model_name,
            "table",
            inputs,
            outputs,
            runtime="table",
        )
    except Exception:
        payload.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest,
            payload,
            model_name,
            inputs,
            outputs,
            written_dtype,
            format_name="Table",
        )
    payload.unlink()
    return manifest


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
    """Write one self-contained, uncompressed native `.fnom` model bundle."""

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
    model_path.unlink()
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
    runtime: str | None = None,
    input_format: str = "numpy",
) -> str:
    input_contract = _contracts(inputs, "inputs")
    output_contract = _contracts(outputs, "outputs")
    dtypes = {tensor.dtype for tensor in (*inputs.values(), *outputs.values())}
    if len(dtypes) != 1:
        raise ValueError("all tensors in one native artifact must share a dtype")
    dtype = dtypes.pop()
    _write_bundle(
        str(manifest),
        str(payload),
        model_name,
        model_format,
        input_contract,
        output_contract,
        dtype,
        list(tree_leaves),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        runtime=runtime,
        input_format=input_format,
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


def _joblib(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    runtime: str = "sklearn",
    input_format: str = "numpy",
    verbose: bool = False,
) -> Path:
    """Export one Joblib model with an explicit resident execution runtime."""

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    if runtime not in {"sklearn", "sklearnex"}:
        raise ValueError("Joblib runtime must be 'sklearn' or 'sklearnex'")
    if input_format not in {"numpy", "pandas", "polars"}:
        raise ValueError("input_format must be 'numpy', 'pandas', or 'polars'")
    if input_format != "numpy" and any(
        tensor.components != 1 for tensor in inputs.values()
    ):
        raise ValueError(
            "pandas and polars input formats currently require scalar input ports"
        )
    manifest, payload, default_name = _destination(path, ".joblib")
    model_name = require_nonempty(name or default_name, "model name")
    temporary = _temporary_for(payload)
    try:
        if write_joblib(model, temporary) == 0:
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
            runtime=runtime,
            input_format=input_format,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest, payload, model_name, inputs, outputs, dtype, format_name="Joblib"
        )
    payload.unlink()
    return manifest


def _compiled(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    verbose: bool = False,
    forest: bool = False,
) -> Path:
    """Export a supported estimator as portable C++ in one FNOM artifact.

    The source is compiled once for the target machine and retained in the
    FoamNordic cache. Supported estimators are linear regressors, fitted
    scikit-learn forests, KNN and gradient boosting, XGBoost, LightGBM, and
    VotingRegressor compositions of supported models.
    """

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    from bindnordic.models.compiled import generate_source

    manifest, payload, default_name = _destination(path, ".cpp")
    model_name = require_nonempty(name or default_name, "model name")
    input_contract = _contracts(inputs, "inputs")
    output_contract = _contracts(outputs, "outputs")
    dtypes = {tensor.dtype for tensor in (*inputs.values(), *outputs.values())}
    if dtypes != {"float64"}:
        raise ValueError("compiled artifacts currently require float64 tensors")
    input_width = sum(width for _, width in input_contract)
    output_width = sum(width for _, width in output_contract)
    if forest:
        from bindnordic.models.forest import encode
        if output_width != 1:
            raise ValueError("forest backend requires one scalar output")
        source = encode(model, input_width)
    else:
        source = generate_source(model, input_width=input_width, output_width=output_width).encode()
    temporary = _temporary_for(payload)
    try:
        temporary.write_bytes(source)
        temporary.replace(payload)
        dtype = _manifest_contract(
            manifest,
            payload,
            model_name,
            "compiled",
            inputs,
            outputs,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            runtime="forest" if forest else "compiled",
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if verbose:
        _display_export(
            manifest, payload, model_name, inputs, outputs, dtype, format_name="Compiled"
        )
    payload.unlink()
    return manifest


def sklearn(
    model: object,
    *,
    path: PathInput,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    name: str | None = None,
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    backend: str = "auto",
    runtime: str = "sklearn",
    input_format: str = "numpy",
    verbose: bool = False,
) -> Path:
    """Export a fitted sklearn-compatible estimator into one FNOM artifact."""

    if backend not in {"auto", "compiled", "joblib", "forest"}:
        raise ValueError("sklearn backend must be 'auto', 'compiled', 'forest', or 'joblib'")
    if runtime not in {"sklearn", "sklearnex"}:
        raise ValueError("sklearn Joblib runtime must be 'sklearn' or 'sklearnex'")
    if input_format not in {"numpy", "pandas", "polars"}:
        raise ValueError("input_format must be 'numpy', 'pandas', or 'polars'")
    from bindnordic.models.compiled import supports

    selected = (
        "compiled"
        if backend == "auto" and input_format == "numpy" and supports(model)
        else backend
    )
    if selected == "auto":
        selected = "joblib"
    arguments = {
        "path": path,
        "inputs": inputs,
        "outputs": outputs,
        "name": name,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "verbose": verbose,
    }
    if selected in {"compiled", "forest"}:
        if input_format != "numpy":
            raise ValueError("compiled sklearn backends require input_format='numpy'")
        if runtime != "sklearn":
            raise ValueError("runtime applies only when the sklearn backend is joblib")
        return _compiled(model, forest=selected == "forest", **arguments)
    return _joblib(model, runtime=runtime, input_format=input_format, **arguments)


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
    manifest, payload, default_name = _destination(path, ".eqx")
    model_name = require_nonempty(name or default_name, "model name")
    temporary = _temporary_for(payload)
    try:
        payload_size, leaves = write_equinox(model, temporary, batched=batched)
        if payload_size == 0:
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
    payload.unlink()
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

    metadata = load_artifact(manifest).as_dict()
    input_scaler = metadata["input_scaler"]
    output_scaler = metadata["output_scaler"]

    table = osm.Table(
        title=f"FoamNordic Model Exported: {name}",
        columns=["Property", "Value"],
        mode="static",
    )
    rows = [
        ("Artifact", manifest.name),
        ("Payload", f"embedded {model.suffix}"),
        ("Format", f"{format_name} + FNOM"),
        ("Dtype", dtype),
        ("Input format", metadata["input_format"]),
        ("Inputs", ", ".join(f"{key}[{value.components}]" for key, value in inputs.items())),
        ("Outputs", ", ".join(f"{key}[{value.components}]" for key, value in outputs.items())),
        ("Input scaler", "none" if input_scaler is None else input_scaler["kind"]),
        ("Output scaler", "none" if output_scaler is None else output_scaler["kind"]),
        ("Compression", "none (load once at worker startup)"),
        ("Payload size", f"{model.stat().st_size / 1024**2:.2f} MiB"),
    ]
    if metadata.get("runtime") is not None:
        rows.insert(3, ("Runtime", metadata["runtime"]))
    for row in rows:
        table.add_row(row)
    table.show()


def __dir__() -> list[str]:
    return sorted(__all__)
