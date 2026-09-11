from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import foamnordic as fno


def test_key_is_immutable_serializable_and_discoverable() -> None:
    key = fno.Random.key(42, scope="rank")
    assert key == fno.Random.Key.from_plan(key.to_plan())
    assert key.scope == "rank"
    assert "uniform" in dir(fno.Random)
    with pytest.raises(FrozenInstanceError):
        key.scope = "global"


def test_split_is_functional_and_reproducible() -> None:
    key = fno.Random.key(42)
    first, second = fno.Random.split(key, 2)
    repeated, _ = fno.Random.split(key, 2)
    assert first == repeated
    assert first != second
    assert (
        fno.Random.uniform(first, shape=(4,)).tolist()
        == fno.Random.uniform(repeated, shape=(4,)).tolist()
    )


def test_basic_distributions_have_expected_shapes() -> None:
    keys = fno.Random.split(fno.Random.key(7), 4)
    assert fno.Random.uniform(keys[0], shape=(2, 3)).shape == (2, 3)
    assert fno.Random.normal(keys[1], shape=5).shape == (5,)
    assert fno.Random.integers(keys[2], 0, 4, shape=3).shape == (3,)
    samples = fno.Random.bernoulli(keys[3], 0.5, shape=8)
    assert samples.dtype == np.dtype(bool)


def test_global_and_rank_invocations_are_distinct_by_contract() -> None:
    from foamnordic.random import invocation

    global_key = fno.Random.key(42, scope="global")
    assert invocation(global_key, "program", 3, 0) == invocation(
        global_key, "program", 3, 1
    )
    rank_key = fno.Random.key(42, scope="rank")
    assert invocation(rank_key, "program", 3, 0) != invocation(
        rank_key, "program", 3, 1
    )


def test_jax_keys_use_the_same_thin_distribution_api() -> None:
    try:
        import jax  # noqa: F401
    except ImportError:
        pytest.skip("JAX is not installed")
    key = fno.Random.to_jax(fno.Random.key(42))
    uniform_key, normal_key = fno.Random.split(key, 2)
    assert fno.Random.uniform(uniform_key, shape=(2,)).shape == (2,)
    assert fno.Random.normal(normal_key, shape=(2,)).shape == (2,)
