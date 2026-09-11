import tempfile
from pathlib import Path

import numpy as np
import pytest
from bindnordic.models.artifact import (
    AffineScaler,
    load,
    normalize_scaler,
    read_payload,
    register_scaler_adapter,
    write_bundle,
)
from bindnordic.models.schema import model_format
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def test_round_trip_preserves_contract_scalers_and_payload():
    values = np.arange(30, dtype=float).reshape(10, 3)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = root / "model.joblib"
        payload.write_bytes(b"payload")
        path = root / "model.fnom"
        write_bundle(
            path,
            payload,
            "model",
            "joblib",
            [("x", 3)],
            [("y", 1)],
            x_scaler=StandardScaler().fit(values),
            y_scaler=MinMaxScaler(clip=True).fit(values[:, :1]),
            runtime="sklearn",
        )
        artifact = load(path)
        assert artifact.name == "model"
        assert artifact.inputs[0].components == 3
        assert artifact.input_scaler.kind == "standard"
        assert artifact.output_scaler.kind == "minmax"
        assert read_payload(path) == b"payload"


def test_invalid_bundle_size_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "broken.fnom"
        path.write_bytes(b"FNOMBIN\0")
        with pytest.raises(ValueError, match="truncated"):
            load(path)


def test_format_schema_centralizes_runtime_and_trust_rules():
    assert not model_format("compiled").is_executable("forest")
    assert model_format("compiled").is_executable("compiled")
    assert model_format("onnx").owns_compute_threads
    assert not model_format("table").owns_compute_threads
    assert not model_format("onnx").concurrent_evaluation
    assert model_format("table").concurrent_evaluation
    with pytest.raises(ValueError, match="runtime"):
        model_format("onnx").validate("python", False)


def test_artifact_reports_derived_execution_capabilities():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = root / "table.bin"
        payload.write_bytes(b"payload")
        path = root / "table.fnom"
        write_bundle(
            path,
            payload,
            "table",
            "table",
            [("x", 1)],
            [("y", 1)],
            runtime="table",
        )
        artifact = load(path)
        assert not artifact.owns_compute_threads
        assert artifact.as_dict()["capabilities"] == {
            "owns_compute_threads": False,
            "concurrent_evaluation": True,
        }


def test_custom_scaler_adapter_reuses_affine_wire_schema():
    class OffsetScaler:
        pass

    register_scaler_adapter(
        OffsetScaler,
        lambda _scaler, features: AffineScaler(
            "function", (1.0,) * features, (2.0,) * features
        ),
    )
    normalized = normalize_scaler(OffsetScaler(), 3)
    assert normalized.bias == (2.0, 2.0, 2.0)
