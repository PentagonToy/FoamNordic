from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

import foamnordic as fno
from foamnordic.core.native_plan import available as native_available


@unittest.skipUnless(native_available(), "nanobind extension is not installed")
class ModelBackendTests(unittest.TestCase):
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
            manifest = fno.export.joblib(
                model,
                path=Path(directory) / "voting.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"nut": fno.Tensor.scalar()},
                x_scaler=scaler,
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
