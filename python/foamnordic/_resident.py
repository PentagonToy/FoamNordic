"""Managed Python model worker behind the native Fjord exchange boundary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

from . import _native


def _manifest(path: Path) -> dict[str, object]:
    return dict(_native.read_model_manifest(str(path)))


def _payload(manifest_path: Path, manifest: Mapping[str, object]) -> Path:
    value = Path(str(manifest["artifact_path"]))
    return value if value.is_absolute() else manifest_path.parent / value


def _output_contract(manifest: Mapping[str, object]) -> tuple[tuple[str, int], ...]:
    return tuple((str(item[0]), int(item[1])) for item in manifest["outputs"])


def _packed_result(value, outputs, rows: int, dtype: str):
    import numpy as np

    if isinstance(value, Mapping):
        pieces = [np.asarray(value[name]).reshape(rows, width) for name, width in outputs]
        result = np.concatenate(pieces, axis=1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(outputs):
            raise ValueError("model output count does not match the FNOM contract")
        pieces = [
            np.asarray(item).reshape(rows, width)
            for item, (_, width) in zip(value, outputs, strict=True)
        ]
        result = np.concatenate(pieces, axis=1)
    else:
        result = np.asarray(value).reshape(rows, sum(width for _, width in outputs))
    return np.ascontiguousarray(result, dtype=dtype)


def _joblib_evaluator(path: Path, outputs):
    try:
        import joblib
        import numpy as np
    except ImportError as error:
        raise ImportError("Joblib is missing from this FoamNordic installation") from error
    model = joblib.load(path, mmap_mode="r")
    predict = getattr(model, "predict", None)
    function = predict if callable(predict) else model
    if not callable(function):
        raise TypeError("Joblib payload must define predict() or be callable")

    def evaluate(buffer, rows, columns, dtype, _exchange_index, _physical_time):
        features = np.frombuffer(buffer, dtype=dtype).reshape(rows, columns)
        return _packed_result(function(features), outputs, rows, dtype)

    return evaluate


def _equinox_evaluator(path: Path, outputs, dtype: str):
    try:
        import cloudpickle
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError as error:
        raise ImportError("Equinox is missing from this FoamNordic installation") from error
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    with path.open("rb") as stream:
        package = cloudpickle.load(stream)
    if not isinstance(package, dict) or package.get("schema") != "foamnordic.equinox/v1":
        raise ValueError("unsupported Equinox payload schema")
    model = package["model"]
    function = model if package.get("batched", False) else jax.vmap(model)
    compiled = jax.jit(function)

    def evaluate(buffer, rows, columns, value_dtype, _exchange_index, _physical_time):
        features = np.frombuffer(buffer, dtype=value_dtype).reshape(rows, columns)
        result = compiled(jnp.asarray(features))
        return _packed_result(result, outputs, rows, value_dtype)

    return evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foamnordic._resident",
        description="Run a Joblib or Equinox model behind native Fjord transport.",
    )
    parser.add_argument("address")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--connections", type=int, default=1)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--no-shm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = _manifest(manifest_path)
    model_format = str(manifest["format"])
    payload = _payload(manifest_path, manifest)
    outputs = _output_contract(manifest)
    dtype = str(manifest["inputs"][0][2])
    if model_format == "joblib":
        evaluator = _joblib_evaluator(payload, outputs)
    elif model_format == "equinox":
        evaluator = _equinox_evaluator(payload, outputs, dtype)
    else:
        raise ValueError(f"managed Python worker does not support {model_format!r}")
    print(f"[FoamNordic] {model_format.title()} model loaded once: {payload}")
    _native.run_python_worker(
        arguments.address,
        str(manifest_path),
        evaluator,
        arguments.connections,
        arguments.ready_file,
        not arguments.no_shm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
