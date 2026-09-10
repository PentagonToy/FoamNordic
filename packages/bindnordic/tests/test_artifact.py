from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from bindnordic.models.artifact import (
    AffineScaler,
    load,
    normalize_scaler,
    read_payload,
    register_scaler_adapter,
    write_bundle,
)
from bindnordic.models.schema import model_format


class ArtifactTests(unittest.TestCase):
    def test_round_trip_preserves_contract_scalers_and_payload(self):
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
            self.assertEqual(artifact.name, "model")
            self.assertEqual(artifact.inputs[0].components, 3)
            self.assertEqual(artifact.input_scaler.kind, "standard")
            self.assertEqual(artifact.output_scaler.kind, "minmax")
            self.assertEqual(read_payload(path), b"payload")

    def test_invalid_bundle_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.fnom"
            path.write_bytes(b"FNOMBIN\0")
            with self.assertRaisesRegex(ValueError, "truncated"):
                load(path)

    def test_format_schema_centralizes_runtime_and_trust_rules(self):
        self.assertFalse(model_format("compiled").is_executable("forest"))
        self.assertTrue(model_format("compiled").is_executable("compiled"))
        with self.assertRaisesRegex(ValueError, "runtime"):
            model_format("onnx").validate("python", False)

    def test_custom_scaler_adapter_reuses_affine_wire_schema(self):
        class OffsetScaler:
            pass

        register_scaler_adapter(
            OffsetScaler,
            lambda _scaler, features: AffineScaler(
                "function", (1.0,) * features, (2.0,) * features
            ),
        )
        normalized = normalize_scaler(OffsetScaler(), 3)
        self.assertEqual(normalized.bias, (2.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
