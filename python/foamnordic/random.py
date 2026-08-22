"""Backend-neutral, reproducible random keys for stochastic field programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Literal


Scope = Literal["global", "rank"]


def _word(value: object) -> int:
    """Map a stable value to one unsigned 32-bit key word."""

    if isinstance(value, bool):
        raise TypeError("random key values must not be booleans")
    if isinstance(value, int):
        return value & 0xFFFFFFFF
    if isinstance(value, (str, bytes)):
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        return int.from_bytes(
            hashlib.blake2s(encoded, digest_size=4).digest(), "little"
        )
    raise TypeError("random key values must be integers, strings, or bytes")


@dataclass(frozen=True, slots=True)
class Key:
    """Immutable root entropy and derivation path.

    ``scope='global'`` shares an invocation key across solver ranks.
    ``scope='rank'`` folds the solver rank into each invocation key.
    """

    entropy: tuple[int, ...]
    path: tuple[int, ...] = ()
    scope: Scope = "global"

    def __post_init__(self) -> None:
        if not self.entropy:
            raise ValueError("random key entropy must not be empty")
        if self.scope not in {"global", "rank"}:
            raise ValueError("random key scope must be global or rank")
        object.__setattr__(self, "entropy", tuple(_word(v) for v in self.entropy))
        object.__setattr__(self, "path", tuple(_word(v) for v in self.path))

    def to_plan(self) -> dict[str, object]:
        return {
            "entropy": list(self.entropy),
            "path": list(self.path),
            "scope": self.scope,
        }

    @classmethod
    def from_plan(cls, value: object) -> "Key":
        if not isinstance(value, dict):
            raise TypeError("random key plan must be a mapping")
        return cls(
            tuple(int(item) for item in value.get("entropy", ())),
            tuple(int(item) for item in value.get("path", ())),
            str(value.get("scope", "global")),  # type: ignore[arg-type]
        )

    def to_json(self) -> str:
        return json.dumps(self.to_plan(), sort_keys=True, separators=(",", ":"))


def key(seed: int = 42, *, scope: Scope = "global") -> Key:
    """Create a backend-neutral root key from one non-negative integer seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random seed must be a non-negative integer")
    words = []
    remaining = seed
    while remaining or len(words) < 2:
        words.append(remaining & 0xFFFFFFFF)
        remaining >>= 32
    return Key(tuple(words), scope=scope)


def _jax_module(value: object):
    """Return JAX when ``value`` is a typed or legacy JAX PRNG key."""

    if isinstance(value, Key):
        return None
    try:
        import jax

        jax.random.key_data(value)
    except (ImportError, TypeError, ValueError):
        return None
    return jax


def fold_in(value, data: object):
    """Derive a key without mutating or consuming the input key."""

    jax = _jax_module(value)
    if jax is not None:
        if isinstance(data, bool) or not isinstance(data, int):
            raise TypeError("JAX fold_in data must be an integer")
        return jax.random.fold_in(value, data)
    if not isinstance(value, Key):
        raise TypeError("fold_in requires a FoamNordic or JAX random key")
    return Key(value.entropy, (*value.path, _word(data)), value.scope)


def split(value, count: int = 2) -> tuple:
    """Return ``count`` independent deterministic child keys."""

    if isinstance(count, bool) or operator.index(count) <= 0:
        raise ValueError("random key split count must be positive")
    jax = _jax_module(value)
    if jax is not None:
        return tuple(jax.random.split(value, count))
    if not isinstance(value, Key):
        raise TypeError("split requires a FoamNordic or JAX random key")
    return tuple(fold_in(fold_in(value, "split"), index) for index in range(count))


def invocation(value: Key, program: str, exchange_index: int, rank: int = 0) -> Key:
    """Derive the key supplied to one field-program invocation."""

    result = fold_in(fold_in(value, program), exchange_index)
    return fold_in(result, rank) if value.scope == "rank" else result


def _generator(value: Key):
    if not isinstance(value, Key):
        raise TypeError("random operations require a FoamNordic Random.Key")
    import numpy as np

    sequence = np.random.SeedSequence(value.entropy, spawn_key=value.path)
    return np.random.default_rng(sequence)


def uniform(
    value,
    low: float = 0.0,
    high: float = 1.0,
    *,
    shape: tuple[int, ...] | int | None = None,
):
    """Draw deterministic uniform values using NumPy-compatible semantics."""

    jax = _jax_module(value)
    if jax is not None:
        return jax.random.uniform(
            value, shape=() if shape is None else shape, minval=low, maxval=high
        )
    return _generator(value).uniform(low, high, size=shape)


def normal(
    value,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    shape: tuple[int, ...] | int | None = None,
):
    """Draw deterministic normal values."""

    jax = _jax_module(value)
    if jax is not None:
        sample = jax.random.normal(value, shape=() if shape is None else shape)
        return sample * std + mean
    return _generator(value).normal(mean, std, size=shape)


def integers(
    value,
    low: int,
    high: int | None = None,
    *,
    shape: tuple[int, ...] | int | None = None,
):
    """Draw deterministic integer values."""

    jax = _jax_module(value)
    if jax is not None:
        minimum, maximum = (0, low) if high is None else (low, high)
        return jax.random.randint(
            value,
            shape=() if shape is None else shape,
            minval=minimum,
            maxval=maximum,
        )
    return _generator(value).integers(low, high, size=shape)


def bernoulli(
    value,
    probability: float = 0.5,
    *,
    shape: tuple[int, ...] | int | None = None,
):
    """Draw deterministic boolean Bernoulli values."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Bernoulli probability must be between zero and one")
    jax = _jax_module(value)
    if jax is not None:
        return jax.random.bernoulli(
            value, probability, shape=() if shape is None else shape
        )
    return _generator(value).random(size=shape) < probability


def to_jax(value: Key):
    """Materialize a FoamNordic key as a JAX typed key."""

    if not isinstance(value, Key):
        raise TypeError("to_jax requires a FoamNordic Random.Key")
    try:
        import jax
    except ImportError as error:
        raise ImportError("JAX is required to materialize a JAX random key") from error
    result = jax.random.key(value.entropy[0])
    for word in (*value.entropy[1:], *value.path):
        result = jax.random.fold_in(result, word)
    return result


class Random:
    """Interactive namespace exposed as ``fno.Random``."""

    Key = Key
    key = staticmethod(key)
    fold_in = staticmethod(fold_in)
    split = staticmethod(split)
    uniform = staticmethod(uniform)
    normal = staticmethod(normal)
    integers = staticmethod(integers)
    bernoulli = staticmethod(bernoulli)
    to_jax = staticmethod(to_jax)

    @staticmethod
    def __dir__() -> list[str]:
        return [
            "Key",
            "bernoulli",
            "fold_in",
            "integers",
            "key",
            "normal",
            "split",
            "to_jax",
            "uniform",
        ]


__all__ = [
    "Key",
    "Random",
    "bernoulli",
    "fold_in",
    "integers",
    "key",
    "normal",
    "split",
    "to_jax",
    "uniform",
]
