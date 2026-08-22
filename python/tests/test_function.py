from __future__ import annotations

import unittest

import numpy as np

from foamnordic.execution.resident import _function_evaluator
import foamnordic as fno


class FunctionOperatorTests(unittest.TestCase):
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
