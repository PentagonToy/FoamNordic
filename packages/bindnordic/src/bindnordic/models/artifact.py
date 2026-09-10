"""Read and write self-contained FNOM model bundles without a solver runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import struct
from typing import Callable, Sequence

from .schema import MODEL_FORMAT_CODES, MODEL_FORMATS, model_format as get_model_format


_MANIFEST_MAGIC = b"FNOMMETA"
_BUNDLE_MAGIC = b"FNOMBIN\0"
_HEADER = struct.Struct("<8sQQQ")
_ALIGNMENT = 64
_MAX_ENTRIES = 65_536
_MAX_STRING = 1024 * 1024
_MAX_MANIFEST = 64 * 1024 * 1024
_DTYPES = {"float32": 1, "float64": 2, "int32": 3, "int64": 4}
_DTYPE_NAMES = {value: name for name, value in _DTYPES.items()}
_DTYPE_BYTES = {"float32": 4, "float64": 8, "int32": 4, "int64": 8}
_SCALERS = {"standard": 1, "minmax": 2, "robust": 3, "maxabs": 4, "function": 5}
_SCALER_NAMES = {value: name for name, value in _SCALERS.items()}


@dataclass(frozen=True, slots=True)
class TensorContract:
    name: str
    components: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model tensor name must not be empty")
        if self.components < 1:
            raise ValueError("model tensor components must be positive")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported model tensor dtype: {self.dtype}")


@dataclass(frozen=True, slots=True)
class AffineScaler:
    kind: str
    gain: tuple[float, ...]
    bias: tuple[float, ...]
    clip_lower: float | None = None
    clip_upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SCALERS:
            raise ValueError(f"unsupported scaler kind: {self.kind}")
        if not self.gain or len(self.gain) != len(self.bias):
            raise ValueError("scaler gain and bias feature counts are invalid")
        if any(not math.isfinite(value) or value == 0 for value in self.gain):
            raise ValueError("scaler gain must contain finite nonzero values")
        if any(not math.isfinite(value) for value in self.bias):
            raise ValueError("scaler bias must contain finite values")
        if (self.clip_lower is None) != (self.clip_upper is None):
            raise ValueError("scaler clipping bounds must be supplied together")
        if self.clip_lower is not None and (
            not math.isfinite(self.clip_lower)
            or not math.isfinite(self.clip_upper)  # type: ignore[arg-type]
            or self.clip_lower > self.clip_upper  # type: ignore[operator]
        ):
            raise ValueError("scaler clipping range is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "gain": list(self.gain),
            "bias": list(self.bias),
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
        }


@dataclass(frozen=True, slots=True)
class TreeLeaf:
    path: str
    dtype: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_count: int

    def __post_init__(self) -> None:
        if not self.path or not self.shape or self.dtype not in _DTYPES:
            raise ValueError("Equinox tree leaf metadata is incomplete")
        elements = 1
        for extent in self.shape:
            if extent < 1:
                raise ValueError("Equinox tree leaf shape is invalid")
            elements *= extent
        if elements * _DTYPE_BYTES[self.dtype] != self.byte_count:
            raise ValueError("Equinox tree leaf byte count is invalid")
        if self.byte_offset < 0:
            raise ValueError("Equinox tree leaf byte offset is invalid")


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    path: Path
    name: str
    format: str
    artifact_path: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    runtime: str | None
    input_scaler: AffineScaler | None
    output_scaler: AffineScaler | None
    tree_leaves: tuple[TreeLeaf, ...]
    payload_offset: int
    payload_size: int

    @property
    def backend(self) -> str:
        return self.format

    @property
    def bundled(self) -> bool:
        return True

    @property
    def executable_payload(self) -> bool:
        return get_model_format(self.format).is_executable(self.runtime)

    @property
    def trust_policy(self) -> str:
        return "explicit-trust-required" if self.executable_payload else "data-only"

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    def validate(self) -> ModelArtifact:
        return load(self.path)

    def as_dict(self) -> dict[str, object]:
        ports = lambda values: [(v.name, v.components, v.dtype) for v in values]
        return {
            "format": self.format,
            "artifact_path": self.artifact_path,
            "name": self.name,
            "inputs": ports(self.inputs),
            "outputs": ports(self.outputs),
            "input_scaler": None if self.input_scaler is None else self.input_scaler.as_dict(),
            "output_scaler": None if self.output_scaler is None else self.output_scaler.as_dict(),
            "runtime": self.runtime,
            "bundled": True,
        }


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def u8(self, value: int) -> None:
        self.data += struct.pack("<B", value)

    def u32(self, value: int) -> None:
        self.data += struct.pack("<I", value)

    def u64(self, value: int) -> None:
        self.data += struct.pack("<Q", value)

    def f64(self, value: float) -> None:
        self.data += struct.pack("<d", value)

    def string(self, value: str) -> None:
        encoded = value.encode()
        if len(encoded) > _MAX_STRING:
            raise ValueError("manifest string exceeds its size limit")
        self.u32(len(encoded))
        self.data += encoded


class _Reader:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.offset = 0

    def raw(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.value):
            raise ValueError("manifest ended before a complete value")
        result = self.value[self.offset:end]
        self.offset = end
        return result

    def unpack(self, kind: str):
        size = struct.calcsize(kind)
        return struct.unpack(kind, self.raw(size))[0]

    def u8(self) -> int:
        return self.unpack("<B")

    def u32(self) -> int:
        return self.unpack("<I")

    def u64(self) -> int:
        return self.unpack("<Q")

    def f64(self) -> float:
        return self.unpack("<d")

    def string(self) -> str:
        count = self.u32()
        if count > _MAX_STRING:
            raise ValueError("manifest string exceeds its size limit")
        return self.raw(count).decode()


def _unique_ports(values: Sequence[TensorContract], label: str) -> None:
    if not values:
        raise ValueError(f"model contract requires {label}")
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise ValueError(f"model {label} names must be unique")


ScalerAdapter = Callable[[object, int], AffineScaler]
_SCALER_ADAPTERS: dict[type[object], ScalerAdapter] = {}


def register_scaler_adapter(
    scaler_type: type[object],
    adapter: ScalerAdapter,
    *,
    replace: bool = False,
) -> None:
    """Register conversion of a fitted scaler to native affine metadata.

    Adapters extend accepted Python scaler classes without changing the FNOM
    wire schema. They must return an :class:`AffineScaler`; nonlinear scaler
    kinds still require an explicit schema and native-runtime addition.
    """

    if scaler_type in _SCALER_ADAPTERS and not replace:
        raise ValueError(f"scaler adapter already registered for {scaler_type.__name__}")
    _SCALER_ADAPTERS[scaler_type] = adapter


def normalize_scaler(scaler: object | None, features: int) -> AffineScaler | None:
    """Normalize one fitted sklearn-compatible affine scaler."""

    if scaler is None:
        return None
    for scaler_type in type(scaler).__mro__:
        adapter = _SCALER_ADAPTERS.get(scaler_type)
        if adapter is not None:
            result = adapter(scaler, features)
            if not isinstance(result, AffineScaler):
                raise TypeError("scaler adapter must return AffineScaler")
            if len(result.gain) != features:
                raise ValueError("scaler adapter feature count does not match model contract")
            return result
    name = scaler.__class__.__name__

    def values(attribute: str, fallback: float) -> tuple[float, ...]:
        value = getattr(scaler, attribute, ...)
        if value is ...:
            raise ValueError(f"fitted scaler is missing {attribute}")
        if value is None:
            return (fallback,) * features
        result = tuple(float(item) for item in value)
        if len(result) != features:
            raise ValueError("scaler feature count does not match model contract")
        return result

    if name == "StandardScaler":
        scale = values("scale_", 1.0)
        gain = tuple(1.0 / value for value in scale)
        center = values("mean_", 0.0)
        return AffineScaler("standard", gain, tuple(-a * b for a, b in zip(center, gain)))
    if name == "MinMaxScaler":
        lower = upper = None
        if bool(getattr(scaler, "clip", False)):
            lower, upper = (float(item) for item in scaler.feature_range)
        return AffineScaler("minmax", values("scale_", 1.0), values("min_", 0.0), lower, upper)
    if name == "MaxAbsScaler":
        scale = values("scale_", 1.0)
        return AffineScaler("maxabs", tuple(1.0 / value for value in scale), (0.0,) * features)
    if name == "RobustScaler":
        scale = values("scale_", 1.0)
        gain = tuple(1.0 / value for value in scale)
        center = values("center_", 0.0)
        return AffineScaler("robust", gain, tuple(-a * b for a, b in zip(center, gain)))
    if name == "FunctionTransformer":
        return _normalize_function_transformer(scaler, features)
    raise ValueError(
        "unsupported fitted scaler: " + name
        + "; expected StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, "
          "or an affine FunctionTransformer"
    )


def _normalize_function_transformer(scaler: object, features: int) -> AffineScaler:
    import numpy as np

    if int(getattr(scaler, "n_features_in_", -1)) != features:
        raise ValueError("FunctionTransformer must be fitted for the model feature count")
    probes = np.zeros((features + 3, features))
    probes[1 : features + 1] = np.eye(features)
    probes[-2] = np.linspace(-1.5, 2.5, features)
    probes[-1] = np.linspace(3.0, -2.0, features)
    result = np.asarray(scaler.transform(probes), dtype=float)
    if result.shape != probes.shape:
        raise ValueError("FunctionTransformer must preserve the feature shape")
    bias = result[0]
    matrix = result[1 : features + 1] - bias
    gain = np.diag(matrix)
    if not np.allclose(matrix, np.diag(gain), rtol=1e-9, atol=1e-11):
        raise ValueError("FunctionTransformer mixes features and cannot be represented natively")
    expected = probes * gain + bias
    if not np.allclose(result, expected, rtol=1e-9, atol=1e-11):
        raise ValueError("FunctionTransformer is nonlinear and cannot be represented natively")
    return AffineScaler("function", tuple(gain), tuple(bias))


def _write_ports(writer: _Writer, ports: Sequence[TensorContract]) -> None:
    if len(ports) > _MAX_ENTRIES:
        raise ValueError("manifest contains too many tensors")
    writer.u32(len(ports))
    for port in ports:
        writer.string(port.name)
        writer.u8(_DTYPES[port.dtype])
        writer.u64(port.components)


def _read_ports(reader: _Reader) -> tuple[TensorContract, ...]:
    count = reader.u32()
    if count > _MAX_ENTRIES:
        raise ValueError("manifest contains too many tensors")
    result = []
    for _ in range(count):
        name = reader.string()
        try:
            dtype = _DTYPE_NAMES[reader.u8()]
        except KeyError as error:
            raise ValueError("manifest tensor dtype is invalid") from error
        result.append(TensorContract(name, reader.u64(), dtype))
    return tuple(result)


def _write_scaler(writer: _Writer, scaler: AffineScaler | None) -> None:
    writer.u8(int(scaler is not None))
    if scaler is None:
        return
    writer.u8(_SCALERS[scaler.kind])
    writer.u32(len(scaler.gain))
    for value in scaler.gain:
        writer.f64(value)
    for value in scaler.bias:
        writer.f64(value)
    writer.u8(int(scaler.clip_lower is not None))
    if scaler.clip_lower is not None:
        writer.f64(scaler.clip_lower)
        writer.f64(scaler.clip_upper)  # type: ignore[arg-type]


def _read_scaler(reader: _Reader) -> AffineScaler | None:
    present = reader.u8()
    if present not in {0, 1}:
        raise ValueError("manifest scaler presence flag is invalid")
    if not present:
        return None
    try:
        kind = _SCALER_NAMES[reader.u8()]
    except KeyError as error:
        raise ValueError("manifest scaler kind is invalid") from error
    count = reader.u32()
    if count < 1 or count > _MAX_ENTRIES:
        raise ValueError("manifest scaler feature count is invalid")
    gain = tuple(reader.f64() for _ in range(count))
    bias = tuple(reader.f64() for _ in range(count))
    clipped = reader.u8()
    if clipped not in {0, 1}:
        raise ValueError("manifest scaler clipping flag is invalid")
    return AffineScaler(
        kind,
        gain,
        bias,
        reader.f64() if clipped else None,
        reader.f64() if clipped else None,
    )


def _encode(
    artifact_path: str,
    name: str,
    model_format: str,
    inputs: Sequence[TensorContract],
    outputs: Sequence[TensorContract],
    leaves: Sequence[TreeLeaf],
    input_scaler: AffineScaler | None,
    output_scaler: AffineScaler | None,
    runtime: str | None,
) -> bytes:
    if not artifact_path or not name or model_format not in MODEL_FORMATS:
        raise ValueError("model artifact metadata is invalid")
    _unique_ports(inputs, "inputs")
    _unique_ports(outputs, "outputs")
    if input_scaler and len(input_scaler.gain) != sum(v.components for v in inputs):
        raise ValueError("input scaler does not match model input features")
    if output_scaler and len(output_scaler.gain) != sum(v.components for v in outputs):
        raise ValueError("output scaler does not match model output features")
    selected_runtime = runtime or ""
    spec = get_model_format(model_format)
    spec.validate(runtime, bool(leaves))
    previous_end = 0
    paths = set()
    for leaf in leaves:
        if leaf.path in paths or leaf.byte_offset < previous_end:
            raise ValueError("Equinox tree leaves overlap or repeat")
        paths.add(leaf.path)
        previous_end = leaf.byte_offset + leaf.byte_count
    writer = _Writer()
    writer.data += _MANIFEST_MAGIC
    writer.u8(spec.code)
    writer.string(artifact_path)
    writer.string(name)
    _write_ports(writer, inputs)
    _write_ports(writer, outputs)
    _write_scaler(writer, input_scaler)
    _write_scaler(writer, output_scaler)
    writer.u32(len(leaves))
    for leaf in leaves:
        writer.string(leaf.path)
        writer.u8(_DTYPES[leaf.dtype])
        writer.u32(len(leaf.shape))
        for extent in leaf.shape:
            writer.u64(extent)
        writer.u64(leaf.byte_offset)
        writer.u64(leaf.byte_count)
    writer.string(selected_runtime)
    return bytes(writer.data)


def _layout(path: Path) -> tuple[int, int, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(_HEADER.size)
    if len(raw) != _HEADER.size:
        raise ValueError(f"FNOM bundle is truncated: {path}")
    magic, manifest_size, payload_offset, payload_size = _HEADER.unpack(raw)
    if magic != _BUNDLE_MAGIC:
        raise ValueError(f"FNOM file is not a self-contained bundle: {path}")
    expected_offset = (_HEADER.size + manifest_size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    if (
        manifest_size < 1
        or manifest_size > _MAX_MANIFEST
        or payload_size < 1
        or payload_offset != expected_offset
        or payload_offset + payload_size != size
    ):
        raise ValueError(f"FNOM bundle size is invalid: {path}")
    return manifest_size, payload_offset, payload_size


def write_bundle(
    path: str | Path,
    payload_path: str | Path,
    name: str,
    model_format: str,
    inputs: Sequence[tuple[str, int]],
    outputs: Sequence[tuple[str, int]],
    dtype: str = "float64",
    tree_leaves: Sequence[tuple[str, str, Sequence[int], int, int]] = (),
    x_scaler: object | None = None,
    y_scaler: object | None = None,
    runtime: str | None = None,
) -> None:
    """Write a byte-compatible FNOM bundle from one payload file."""

    destination = Path(path).expanduser().resolve()
    payload = Path(payload_path).expanduser().resolve()
    if not payload.is_file() or payload.stat().st_size < 1:
        raise ValueError("model payload must not be empty")
    in_ports = tuple(TensorContract(n, c, dtype) for n, c in inputs)
    out_ports = tuple(TensorContract(n, c, dtype) for n, c in outputs)
    leaves = tuple(TreeLeaf(p, d, tuple(s), o, c) for p, d, s, o, c in tree_leaves)
    manifest = _encode(
        payload.name,
        name,
        model_format,
        in_ports,
        out_ports,
        leaves,
        normalize_scaler(x_scaler, sum(p.components for p in in_ports)),
        normalize_scaler(y_scaler, sum(p.components for p in out_ports)),
        runtime,
    )
    payload_offset = (_HEADER.size + len(manifest) + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.bundle.tmp"
    try:
        with temporary.open("wb") as output, payload.open("rb") as source:
            output.write(_HEADER.pack(_BUNDLE_MAGIC, len(manifest), payload_offset, payload.stat().st_size))
            output.write(manifest)
            output.write(b"\0" * (payload_offset - _HEADER.size - len(manifest)))
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load(path: str | Path) -> ModelArtifact:
    selected = Path(path).expanduser().resolve()
    if selected.suffix.lower() != ".fnom":
        raise ValueError("model artifacts must use the .fnom extension")
    if not selected.is_file():
        raise FileNotFoundError(f"FNOM artifact does not exist: {selected}")
    manifest_size, payload_offset, payload_size = _layout(selected)
    with selected.open("rb") as stream:
        stream.seek(_HEADER.size)
        value = stream.read(manifest_size)
    reader = _Reader(value)
    if reader.raw(8) != _MANIFEST_MAGIC:
        raise ValueError("manifest magic is invalid")
    try:
        model_format = MODEL_FORMAT_CODES[reader.u8()].name
    except KeyError as error:
        raise ValueError("manifest model format is invalid") from error
    artifact_path = reader.string()
    name = reader.string()
    inputs = _read_ports(reader)
    outputs = _read_ports(reader)
    input_scaler = _read_scaler(reader)
    output_scaler = _read_scaler(reader)
    count = reader.u32()
    if count > _MAX_ENTRIES:
        raise ValueError("manifest contains too many tree leaves")
    leaves = []
    for _ in range(count):
        leaf_path = reader.string()
        try:
            leaf_dtype = _DTYPE_NAMES[reader.u8()]
        except KeyError as error:
            raise ValueError("manifest tree leaf dtype is invalid") from error
        rank = reader.u32()
        if rank < 1 or rank > _MAX_ENTRIES:
            raise ValueError("manifest tree leaf rank is invalid")
        shape = tuple(reader.u64() for _ in range(rank))
        leaves.append(TreeLeaf(leaf_path, leaf_dtype, shape, reader.u64(), reader.u64()))
    runtime = reader.string() or None
    if reader.offset != len(value):
        raise ValueError("manifest contains trailing data")
    _encode(artifact_path, name, model_format, inputs, outputs, leaves, input_scaler, output_scaler, runtime)
    return ModelArtifact(
        selected, name, model_format, artifact_path, inputs, outputs, runtime,
        input_scaler, output_scaler, tuple(leaves), payload_offset, payload_size,
    )


def payload_region(path: str | Path) -> tuple[int, int]:
    _, offset, size = _layout(Path(path).expanduser().resolve())
    return offset, size


def read_payload(path: str | Path) -> bytes:
    selected = Path(path).expanduser().resolve()
    offset, size = payload_region(selected)
    with selected.open("rb") as stream:
        stream.seek(offset)
        value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"could not read FNOM payload: {selected}")
    return value


def extract_payload(path: str | Path, destination: str | Path) -> None:
    selected = Path(path).expanduser().resolve()
    offset, size = payload_region(selected)
    remaining = size
    with selected.open("rb") as source, Path(destination).open("wb") as output:
        source.seek(offset)
        while remaining:
            chunk = source.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"could not read FNOM payload: {selected}")
            output.write(chunk)
            remaining -= len(chunk)


__all__ = [
    "AffineScaler", "ModelArtifact", "TensorContract", "TreeLeaf",
    "extract_payload", "load", "normalize_scaler", "payload_region",
    "register_scaler_adapter",
    "read_payload", "write_bundle",
]
