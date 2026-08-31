from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import foamnordic as fno
from foamnordic.core.native_plan import available as native_available


@unittest.skipUnless(native_available(), "nanobind extension is not installed")
class ModelBackendTests(unittest.TestCase):
    def _compiled_prediction(self, model, features, root: Path, *, threads: int = 2):
        from foamnordic.execution.resident import _compiled_evaluator

        artifact = fno.export.sklearn(
            model,
            path=root / "model.fnom",
            inputs={"features": fno.Tensor.vector(components=features.shape[1])},
            outputs={"prediction": fno.Tensor.scalar()},
        )
        with patch.dict(
            "os.environ", {"FOAMNORDIC_COMPILED_CACHE": str(root / "cache")}
        ):
            evaluator = _compiled_evaluator(
                BytesIO(fno._native.read_model_payload(str(artifact))),
                (("prediction", 1),),
                "float64",
                threads=threads,
            )
        return evaluator(
            memoryview(features), len(features), features.shape[1], "float64", 0, 0.0
        )[:, 0]

    def test_sklearn_knn_compiled_round_trip(self) -> None:
        from sklearn.neighbors import KNeighborsRegressor

        rng = np.random.default_rng(40)
        training = rng.normal(size=(128, 4))
        targets = np.sin(training[:, 0]) + training[:, 1]
        features = np.ascontiguousarray(rng.normal(size=(512, 4)))
        for weights in ("uniform", "distance"):
            model = KNeighborsRegressor(
                n_neighbors=5, weights=weights, p=2
            ).fit(training, targets)
            with tempfile.TemporaryDirectory() as directory:
                actual = self._compiled_prediction(model, features, Path(directory))
            np.testing.assert_allclose(
                actual, model.predict(features), rtol=1.0e-14, atol=1.0e-14
            )

    def test_sklearn_gradient_boosting_compiled_round_trip(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(41)
        training = rng.normal(size=(256, 4))
        targets = np.sin(training[:, 0]) + 0.25 * training[:, 1]
        features = np.ascontiguousarray(rng.normal(size=(512, 4)))
        model = GradientBoostingRegressor(
            n_estimators=12, max_depth=3, random_state=41
        ).fit(training, targets)
        with tempfile.TemporaryDirectory() as directory:
            actual = self._compiled_prediction(model, features, Path(directory))
        np.testing.assert_allclose(
            actual, model.predict(features), rtol=1.0e-14, atol=1.0e-14
        )

    def test_sklearn_extra_trees_knn_voting_compiled_round_trip(self) -> None:
        from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
        from sklearn.neighbors import KNeighborsRegressor

        rng = np.random.default_rng(42)
        training = rng.normal(size=(256, 4))
        targets = np.sin(training[:, 0]) + 0.25 * training[:, 1]
        features = np.ascontiguousarray(rng.normal(size=(512, 4)))
        model = VotingRegressor(
            estimators=[
                ("trees", ExtraTreesRegressor(n_estimators=8, random_state=42)),
                ("neighbors", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ],
            weights=[0.5, 0.5],
        ).fit(training, targets)
        with tempfile.TemporaryDirectory() as directory:
            actual = self._compiled_prediction(model, features, Path(directory))
        np.testing.assert_allclose(
            actual, model.predict(features), rtol=1.0e-14, atol=1.0e-14
        )

    def test_xgboost_compiled_round_trip(self) -> None:
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError):
            self.skipTest("XGBoost is not installed")

        rng = np.random.default_rng(44)
        training = rng.normal(size=(256, 4))
        targets = np.sin(training[:, 0]) + 0.25 * training[:, 1]
        features = np.ascontiguousarray(rng.normal(size=(512, 4)))
        features[0, 0] = np.nan
        model = XGBRegressor(
            n_estimators=12, max_depth=3, n_jobs=1, random_state=44, verbosity=0
        ).fit(training, targets)
        with tempfile.TemporaryDirectory() as directory:
            actual = self._compiled_prediction(model, features, Path(directory))
        np.testing.assert_allclose(
            actual, model.predict(features), rtol=2.0e-6, atol=2.0e-6
        )

    def test_lightgbm_compiled_round_trip(self) -> None:
        try:
            from lightgbm import LGBMRegressor
        except (ImportError, OSError):
            self.skipTest("LightGBM is not installed")

        rng = np.random.default_rng(45)
        training = rng.normal(size=(256, 4))
        targets = np.sin(training[:, 0]) + 0.25 * training[:, 1]
        features = np.ascontiguousarray(rng.normal(size=(512, 4)))
        features[0, 0] = np.nan
        model = LGBMRegressor(
            n_estimators=12, max_depth=3, n_jobs=1, random_state=45, verbosity=-1
        ).fit(training, targets)
        with tempfile.TemporaryDirectory() as directory:
            actual = self._compiled_prediction(model, features, Path(directory))
        np.testing.assert_allclose(
            actual, model.predict(features), rtol=1.0e-12, atol=1.0e-12
        )

    def test_sklearn_extra_trees_compiled_round_trip(self) -> None:
        from sklearn.ensemble import ExtraTreesRegressor

        from foamnordic.execution.resident import _compiled_evaluator

        rng = np.random.default_rng(42)
        features = rng.normal(size=(512, 4))
        targets = np.sin(features[:, 0]) + 0.25 * features[:, 1]
        model = ExtraTreesRegressor(
            n_estimators=8, max_depth=8, random_state=42
        ).fit(features, targets)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = fno.export.sklearn(
                model,
                path=root / "trees.fnom",
                inputs={"features": fno.Tensor.vector(components=4)},
                outputs={"prediction": fno.Tensor.scalar()},
            )
            metadata = fno._native.read_model_manifest(str(artifact))
            self.assertEqual(metadata["format"], "compiled")
            self.assertEqual(metadata["runtime"], "cpp-v1")
            with patch.dict(
                "os.environ", {"FOAMNORDIC_COMPILED_CACHE": str(root / "cache")}
            ):
                first = _compiled_evaluator(
                    BytesIO(fno._native.read_model_payload(str(artifact))),
                    (("prediction", 1),),
                    "float64",
                    threads=2,
                )
                second = _compiled_evaluator(
                    BytesIO(fno._native.read_model_payload(str(artifact))),
                    (("prediction", 1),),
                    "float64",
                    threads=2,
                )
            actual = second(
                memoryview(np.ascontiguousarray(features)),
                len(features),
                features.shape[1],
                "float64",
                0,
                0.0,
            )
            self.assertFalse(first.foamnordic_cache_hit)
            self.assertTrue(second.foamnordic_cache_hit)
            np.testing.assert_allclose(
                actual[:, 0], model.predict(features), rtol=1.0e-14, atol=1.0e-14
            )

    def test_sklearn_extra_trees_compiled_large_batch(self) -> None:
        from sklearn.ensemble import ExtraTreesRegressor

        from foamnordic.execution.resident import _compiled_evaluator

        rng = np.random.default_rng(43)
        training = rng.normal(size=(512, 4))
        targets = np.sin(training[:, 0]) + 0.25 * training[:, 1]
        model = ExtraTreesRegressor(
            n_estimators=8, max_depth=8, random_state=43
        ).fit(training, targets)
        features = np.ascontiguousarray(rng.normal(size=(32768, 4)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = fno.export.sklearn(
                model,
                path=root / "trees.fnom",
                inputs={"features": fno.Tensor.vector(components=4)},
                outputs={"prediction": fno.Tensor.scalar()},
            )
            with patch.dict(
                "os.environ", {"FOAMNORDIC_COMPILED_CACHE": str(root / "cache")}
            ):
                evaluator = _compiled_evaluator(
                    BytesIO(fno._native.read_model_payload(str(artifact))),
                    (("prediction", 1),),
                    "float64",
                    threads=2,
                )
            actual = evaluator(
                memoryview(features), len(features), 4, "float64", 0, 0.0
            )
            np.testing.assert_allclose(
                actual[:, 0], model.predict(features), rtol=1.0e-14, atol=1.0e-14
            )

    def test_sklearn_voting_regressor_joblib_round_trip(self) -> None:
        from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        features = np.asarray(
            [[-2.0, 1.0], [-1.0, 2.0], [0.0, 0.0], [1.0, 3.0], [2.0, 4.0], [3.0, 1.0]],
            dtype=np.float64,
        )
        targets = 0.5 * features[:, 0] - 0.25 * features[:, 1]
        scaler = StandardScaler().fit(features)
        scaled = scaler.transform(features)
        model = VotingRegressor(
            estimators=[
                ("trees", ExtraTreesRegressor(n_estimators=4, random_state=42)),
                ("neighbors", KNeighborsRegressor(n_neighbors=2)),
            ],
            weights=[0.7, 0.3],
        ).fit(scaled, targets)

        from foamnordic.execution.resident import _joblib_evaluator

        with tempfile.TemporaryDirectory() as directory:
            manifest = fno.export.sklearn(
                model,
                path=Path(directory) / "voting.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"nut": fno.Tensor.scalar()},
                x_scaler=scaler,
                backend="joblib",
            )
            metadata = fno._native.read_model_manifest(str(manifest))
            self.assertEqual(metadata["input_scaler"]["kind"], "standard")
            evaluator = _joblib_evaluator(
                BytesIO(fno._native.read_model_payload(str(manifest))),
                (("nut", 1),),
            )
            actual = evaluator(
                memoryview(np.ascontiguousarray(scaled)),
                len(scaled),
                2,
                "float64",
                0,
                0.0,
            )
            np.testing.assert_allclose(actual[:, 0], model.predict(scaled))

    def test_equinox_small_mlp_round_trip(self) -> None:
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        model = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=4,
            depth=1,
            key=jax.random.PRNGKey(42),
        )
        values = np.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)

        from foamnordic.execution.resident import _equinox_evaluator

        with tempfile.TemporaryDirectory() as directory:
            manifest = fno.export.equinox(
                model,
                path=Path(directory) / "mlp.fnom",
                inputs={"features": fno.Tensor.vector(components=2, dtype="float32")},
                outputs={"nut": fno.Tensor.scalar(dtype="float32")},
            )
            evaluator = _equinox_evaluator(
                BytesIO(fno._native.read_model_payload(str(manifest))),
                (("nut", 1),),
                "float32",
            )
            actual = evaluator(memoryview(values), 2, 2, "float32", 0, 0.0)
            expected = jax.vmap(model)(jnp.asarray(values))
            np.testing.assert_allclose(actual, expected, rtol=1.0e-6)

    def test_equinox_key_is_derived_from_seed_and_exchange_index(self) -> None:
        import cloudpickle
        import jax
        import jax.numpy as jnp

        def stochastic(value, *, key):
            return value[:1] + jax.random.uniform(key, (1,), dtype=value.dtype)

        from foamnordic.execution.resident import _equinox_evaluator

        values = np.asarray([[0.0], [1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "stochastic.eqx"
            with payload.open("wb") as stream:
                cloudpickle.dump(
                    {
                        "schema": "foamnordic.equinox/v1",
                        "model": stochastic,
                        "batched": False,
                    },
                    stream,
                )
            first = _equinox_evaluator(
                payload, (("prediction", 1),), "float32", seed=42
            )
            second = _equinox_evaluator(
                payload, (("prediction", 1),), "float32", seed=42
            )
            first_zero = first(memoryview(values), 2, 1, "float32", 0, 0.0)
            second_zero = second(memoryview(values), 2, 1, "float32", 0, 0.0)
            first_one = first(memoryview(values), 2, 1, "float32", 1, 0.1)
            np.testing.assert_allclose(first_zero, second_zero)
            self.assertFalse(np.allclose(first_zero, first_one))

    def test_real_onnx_graph_round_trip(self) -> None:
        import onnx
        from onnx import TensorProto, helper
        from onnx.reference import ReferenceEvaluator

        weight = helper.make_tensor("weight", TensorProto.FLOAT, [2, 1], [2.0, -1.0])
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [0.5])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["features", "weight", "bias"], ["nut"])],
            "foamnordic-linear-smoke",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("nut", TensorProto.FLOAT, [None, 1])],
            [weight, bias],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=9,
        )
        onnx.checker.check_model(model)

        with tempfile.TemporaryDirectory() as directory:
            manifest = fno.export.onnx(
                model,
                path=Path(directory) / "linear.fnom",
                inputs={"features": fno.Tensor.vector(components=2, dtype="float32")},
                outputs={"nut": fno.Tensor.scalar(dtype="float32")},
            )
            loaded = onnx.load_from_string(
                fno._native.read_model_payload(str(manifest))
            )
            values = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
            actual = ReferenceEvaluator(loaded).run(None, {"features": values})[0]
            np.testing.assert_allclose(actual, [[-0.5], [0.5]], rtol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
