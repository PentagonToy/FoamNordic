from __future__ import annotations

from pathlib import Path
import inspect
import tempfile
import unittest
from unittest.mock import patch

import foamnordic as fno
from foamnordic.core.native_plan import available as native_available


class _LinearPrediction:
    def __init__(self):
        import numpy as np

        self.weights = np.asarray([2.0, 1.0], dtype=np.float64)

    def predict(self, values):
        return (values @ self.weights).reshape(-1, 1)


class ExportTests(unittest.TestCase):
    def test_export_help_is_backend_discoverable(self) -> None:
        self.assertEqual(dir(fno.export), ["Tensor", "equinox", "joblib", "onnx"])

    def test_verbose_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "boolean"):
            fno.export.onnx(
                b"onnx",
                path="model.fnom",
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                verbose="yes",
            )

    def test_scalers_default_to_none_for_every_exporter(self) -> None:
        for exporter in (fno.export.onnx, fno.export.joblib, fno.export.equinox):
            parameters = inspect.signature(exporter).parameters
            self.assertIsNone(parameters["x_scaler"].default)
            self.assertIsNone(parameters["y_scaler"].default)

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_existing_onnx_payload_exports_native_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fno.export.onnx(
                b"onnx-payload",
                path=root / "reaction-rate.fnom",
                name="reaction-rate",
                inputs={
                    "c_tilde": fno.Tensor.scalar(),
                    "c_var": fno.Tensor.scalar(),
                    "T_tilde": fno.Tensor.scalar(),
                },
                outputs={"omega_c": fno.Tensor.scalar()},
            )
            self.assertEqual(manifest.read_bytes()[:8], b"FNOMAN1\0")
            self.assertEqual(
                (root / "reaction-rate.onnx").read_bytes(), b"onnx-payload"
            )
            self.assertLess(manifest.stat().st_size, 1024)

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_path_backed_payload_is_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large-voting-regressor.onnx"
            source.write_bytes(b"model" * 1024)
            manifest = fno.export.onnx(
                source,
                path=root / "voting.fnom",
                inputs={"features": fno.Tensor.vector(components=3)},
                outputs={"prediction": fno.Tensor.scalar()},
            )
            self.assertEqual(
                manifest.with_suffix(".onnx").read_bytes(), source.read_bytes()
            )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_verbose_true_displays_export_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "foamnordic.export._display_export"
        ) as display:
            fno.export.onnx(
                b"onnx-payload",
                path=Path(directory) / "model.fnom",
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                verbose=True,
            )
        display.assert_called_once()

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_fitted_sklearn_scalers_are_normalized_into_fnom(self) -> None:
        try:
            import numpy as np
            from sklearn.preprocessing import (
                FunctionTransformer,
                MaxAbsScaler,
                MinMaxScaler,
                RobustScaler,
                StandardScaler,
            )
        except ImportError:
            self.skipTest("scikit-learn test dependencies are unavailable")

        features = np.asarray(
            [[0.0, -2.0], [2.0, 4.0], [4.0, 10.0]], dtype=np.float64
        )
        targets = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)
        scalers = (
            (StandardScaler().fit(features), "standard"),
            (MinMaxScaler(feature_range=(-1.0, 1.0), clip=True).fit(features), "minmax"),
            (MaxAbsScaler().fit(features), "maxabs"),
            (RobustScaler().fit(features), "robust"),
            (
                FunctionTransformer(
                    func=lambda values: values * np.asarray([2.0, -4.0])
                    + np.asarray([3.0, 5.0]),
                    inverse_func=lambda values: (values - np.asarray([3.0, 5.0]))
                    / np.asarray([2.0, -4.0]),
                ).fit(features),
                "function",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (x_scaler, expected_kind) in enumerate(scalers):
                manifest = fno.export.onnx(
                    b"onnx-payload",
                    path=root / f"scaled-{index}.fnom",
                    inputs={"features": fno.Tensor.vector(components=2)},
                    outputs={"prediction": fno.Tensor.scalar()},
                    x_scaler=x_scaler,
                    y_scaler=StandardScaler().fit(targets),
                )
                metadata = fno._native.read_model_manifest(str(manifest))
                self.assertEqual(metadata["input_scaler"]["kind"], expected_kind)
                self.assertEqual(metadata["output_scaler"]["kind"], "standard")
                self.assertEqual(len(metadata["input_scaler"]["gain"]), 2)
                self.assertEqual(len(metadata["output_scaler"]["gain"]), 1)

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_nonlinear_function_transformer_is_rejected(self) -> None:
        try:
            import numpy as np
            from sklearn.preprocessing import FunctionTransformer
        except ImportError:
            self.skipTest("scikit-learn test dependencies are unavailable")
        scaler = FunctionTransformer(np.square).fit(
            np.asarray([[0.0, 1.0], [1.0, 2.0]])
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "nonlinear"
        ):
            fno.export.onnx(
                b"onnx-payload",
                path=Path(directory) / "nonlinear.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"prediction": fno.Tensor.scalar()},
                x_scaler=scaler,
            )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_scaler_feature_mismatch_fails_during_export(self) -> None:
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.skipTest("scikit-learn test dependencies are unavailable")
        scaler = StandardScaler().fit(np.asarray([[0.0], [1.0]]))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "feature"
        ):
            fno.export.onnx(
                b"onnx-payload",
                path=Path(directory) / "mismatch.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"prediction": fno.Tensor.scalar()},
                x_scaler=scaler,
            )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_joblib_model_is_path_backed_and_mmap_loadable(self) -> None:
        try:
            import joblib
            import numpy as np
        except ImportError:
            self.skipTest("Joblib test dependencies are unavailable")
        from foamnordic.execution.resident import _joblib_evaluator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fno.export.joblib(
                _LinearPrediction(),
                path=root / "voting.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"prediction": fno.Tensor.scalar()},
            )
            metadata = fno._native.read_model_manifest(str(manifest))
            self.assertEqual(metadata["format"], "joblib")
            payload = root / metadata["artifact_path"]
            self.assertTrue(payload.is_file())
            loaded = joblib.load(payload, mmap_mode="r")
            self.assertIsInstance(loaded.weights, np.memmap)
            values = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)
            evaluator = _joblib_evaluator(payload, (("prediction", 1),))
            result = evaluator(memoryview(values), 2, 2, "float64", 0, 0.0)
            np.testing.assert_allclose(result, [[5.0], [8.0]])

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_equinox_model_exports_tree_metadata_and_evaluates(self) -> None:
        try:
            import equinox as eqx
            import jax
            import jax.numpy as jnp
            import numpy as np
        except ImportError:
            self.skipTest("Equinox test dependencies are unavailable")
        from foamnordic.execution.resident import _equinox_evaluator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(42))
            manifest = fno.export.equinox(
                model,
                path=root / "closure.fnom",
                inputs={"features": fno.Tensor.vector(components=2, dtype="float32")},
                outputs={"prediction": fno.Tensor.scalar(dtype="float32")},
            )
            metadata = fno._native.read_model_manifest(str(manifest))
            self.assertEqual(metadata["format"], "equinox")
            payload = root / metadata["artifact_path"]
            evaluator = _equinox_evaluator(
                payload, (("prediction", 1),), "float32"
            )
            values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            result = evaluator(memoryview(values), 2, 2, "float32", 0, 0.0)
            expected = jax.vmap(model)(jnp.asarray(values))
            np.testing.assert_allclose(result, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
