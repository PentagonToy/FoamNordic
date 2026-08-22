from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

import foamnordic as fno


class RandomTests(unittest.TestCase):
    def test_key_is_immutable_serializable_and_discoverable(self) -> None:
        key = fno.Random.key(42, scope="rank")
        self.assertEqual(key, fno.Random.Key.from_plan(key.to_plan()))
        self.assertEqual(key.scope, "rank")
        self.assertIn("uniform", dir(fno.Random))
        with self.assertRaises(FrozenInstanceError):
            key.scope = "global"

    def test_split_is_functional_and_reproducible(self) -> None:
        key = fno.Random.key(42)
        first, second = fno.Random.split(key, 2)
        repeated, _ = fno.Random.split(key, 2)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(
            fno.Random.uniform(first, shape=(4,)).tolist(),
            fno.Random.uniform(repeated, shape=(4,)).tolist(),
        )

    def test_basic_distributions_have_expected_shapes(self) -> None:
        keys = fno.Random.split(fno.Random.key(7), 4)
        self.assertEqual(fno.Random.uniform(keys[0], shape=(2, 3)).shape, (2, 3))
        self.assertEqual(fno.Random.normal(keys[1], shape=5).shape, (5,))
        self.assertEqual(fno.Random.integers(keys[2], 0, 4, shape=3).shape, (3,))
        samples = fno.Random.bernoulli(keys[3], 0.5, shape=8)
        self.assertEqual(samples.dtype, np.dtype(bool))

    def test_global_and_rank_invocations_are_distinct_by_contract(self) -> None:
        from foamnordic.random import invocation

        global_key = fno.Random.key(42, scope="global")
        self.assertEqual(
            invocation(global_key, "program", 3, 0),
            invocation(global_key, "program", 3, 1),
        )
        rank_key = fno.Random.key(42, scope="rank")
        self.assertNotEqual(
            invocation(rank_key, "program", 3, 0),
            invocation(rank_key, "program", 3, 1),
        )

    def test_jax_keys_use_the_same_thin_distribution_api(self) -> None:
        try:
            import jax
        except ImportError:
            self.skipTest("JAX is not installed")
        key = fno.Random.to_jax(fno.Random.key(42))
        uniform_key, normal_key = fno.Random.split(key, 2)
        self.assertEqual(fno.Random.uniform(uniform_key, shape=(2,)).shape, (2,))
        self.assertEqual(fno.Random.normal(normal_key, shape=(2,)).shape, (2,))


if __name__ == "__main__":
    unittest.main()
