from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from foamnordic.execution.resident import _function_evaluator
import foamnordic as fno


class FunctionOperatorTests(unittest.TestCase):
    def test_unused_random_state_is_not_constructed(self) -> None:
        package = {
            "schema": "foamnordic.function/v1",
            "function": lambda value: value + 1.0,
            "inputs": ("value",),
            "input_widths": (1,),
            "outputs": ("result",),
            "seed": 42,
        }
        values = np.asarray([[1.0], [2.0]], dtype=np.float64)
        evaluator = _function_evaluator(package, (("result", 1),))

        with patch("numpy.random.default_rng") as default_rng:
            actual = evaluator(memoryview(values), 2, 1, "float64", 3, 0.1)

        default_rng.assert_not_called()
        np.testing.assert_array_equal(actual[:, 0], np.asarray([2.0, 3.0]))

    def test_tensor_port_is_exposed_in_physical_matrix_shape(self) -> None:
        observed = None

        def keqn(k, velocity_grad, delta):
            nonlocal observed
            observed = velocity_grad.shape
            symmetric = fno.Math.symm(velocity_grad)
            production = fno.Math.ddot(velocity_grad, symmetric)
            return k + delta * production

        package = {
            "schema": "foamnordic.function/v1",
            "function": keqn,
            "inputs": ("k", "velocity_grad", "delta"),
            "input_widths": (1, 9, 1),
            "input_layouts": (
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
                {
                    "kind": "tensor",
                    "physical_shape": [3, 3],
                    "transport_width": 9,
                },
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
            ),
            "outputs": ("nut",),
            "output_layouts": (
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
            ),
            "seed": 42,
        }
        rows = 2
        k = np.asarray([[1.0], [2.0]])
        gradient = np.arange(18, dtype=np.float64).reshape(rows, 9)
        delta = np.asarray([[0.5], [0.25]])
        values = np.concatenate((k, gradient, delta), axis=1)
        evaluator = _function_evaluator(package, (("nut", 1),))

        actual = evaluator(memoryview(values), rows, 11, "float64", 0, 0.0)

        self.assertEqual(observed, (rows, 3, 3))
        expected_gradient = gradient.reshape(rows, 3, 3)
        expected = k[:, 0] + delta[:, 0] * np.sum(
            expected_gradient
            * 0.5
            * (expected_gradient + expected_gradient.swapaxes(-1, -2)),
            axis=(-2, -1),
        )
        np.testing.assert_allclose(actual[:, 0], expected)

    def test_symmetric_tensor_uses_matrix_shape_at_function_boundary(self) -> None:
        def identity(stress):
            self.assertEqual(stress.shape, (2, 3, 3))
            np.testing.assert_array_equal(stress, stress.swapaxes(-1, -2))
            return stress

        layout = {
            "kind": "symm_tensor",
            "physical_shape": [3, 3],
            "transport_width": 6,
        }
        package = {
            "schema": "foamnordic.function/v1",
            "function": identity,
            "inputs": ("stress",),
            "input_widths": (6,),
            "input_layouts": (layout,),
            "outputs": ("stress",),
            "output_layouts": (layout,),
            "seed": 42,
        }
        values = np.arange(12, dtype=np.float64).reshape(2, 6)
        evaluator = _function_evaluator(package, (("stress", 6),))

        actual = evaluator(memoryview(values), 2, 6, "float64", 0, 0.0)

        np.testing.assert_array_equal(actual, values)

    def test_named_fields_and_seeded_rng_are_deterministic(self) -> None:
        def perturb(velocity, *, rng, exchange_index, physical_time):
            scale = rng.uniform(0.995, 1.005, size=(velocity.shape[0], 1))
            return {"updated": velocity * scale + 0 * exchange_index + 0 * physical_time}

        package = {
            "schema": "foamnordic.function/v1",
            "function": perturb,
            "inputs": ("velocity",),
            "input_widths": (3,),
            "outputs": ("updated",),
            "seed": 42,
        }
        outputs = (("updated", 3),)
        values = np.arange(12, dtype=np.float64).reshape(4, 3)
        evaluator = _function_evaluator(package, outputs)
        first = evaluator(memoryview(values), 4, 3, "float64", 7, 0.1)
        second = evaluator(memoryview(values), 4, 3, "float64", 7, 0.1)
        third = evaluator(memoryview(values), 4, 3, "float64", 8, 0.2)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(first.shape, (4, 3))

    def test_scalar_and_vector_ports_are_unpacked_by_logical_name(self) -> None:
        def predict(x_coordinate, pressure):
            return np.column_stack((x_coordinate, pressure, x_coordinate + pressure))

        package = {
            "schema": "foamnordic.function/v1",
            "function": predict,
            "inputs": ("x_coordinate", "pressure"),
            "input_widths": (1, 1),
            "outputs": ("velocity",),
            "seed": 42,
        }
        evaluator = _function_evaluator(package, (("velocity", 3),))
        values = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)
        actual = evaluator(memoryview(values), 2, 2, "float64", 0, 0.0)
        expected = np.asarray([[1.0, 3.0, 4.0], [2.0, 4.0, 6.0]])
        np.testing.assert_array_equal(actual, expected)

    def test_random_key_scope_controls_rank_derivation(self) -> None:
        def perturb(velocity, *, key, rank):
            scale_key, noise_key = fno.Random.split(key, 2)
            scale = fno.Random.uniform(scale_key, low=0.995, high=1.005)
            noise = fno.Random.normal(
                noise_key, shape=velocity.shape, std=1.0e-6
            )
            return {"updated": velocity * scale + noise + 0 * rank}

        values = np.ones((3, 3), dtype=np.float64)
        base = {
            "schema": "foamnordic.function/v1",
            "function": perturb,
            "program": "perturbVelocity",
            "inputs": ("velocity",),
            "input_widths": (3,),
            "outputs": ("updated",),
        }
        outputs = (("updated", 3),)

        global_evaluator = _function_evaluator(
            {**base, "key": fno.Random.key(42, scope="global").to_plan()},
            outputs,
        )
        global_rank_zero = global_evaluator(
            memoryview(values), 3, 3, "float64", 7, 0.1, 0
        )
        global_rank_one = global_evaluator(
            memoryview(values), 3, 3, "float64", 7, 0.1, 1
        )
        np.testing.assert_array_equal(global_rank_zero, global_rank_one)

        rank_evaluator = _function_evaluator(
            {**base, "key": fno.Random.key(42, scope="rank").to_plan()},
            outputs,
        )
        rank_zero = rank_evaluator(
            memoryview(values), 3, 3, "float64", 7, 0.1, 0
        )
        rank_one = rank_evaluator(
            memoryview(values), 3, 3, "float64", 7, 0.1, 1
        )
        self.assertFalse(np.array_equal(rank_zero, rank_one))


if __name__ == "__main__":
    unittest.main()
