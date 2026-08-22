"""Small deterministic combustion oracles used by acceptance fixtures.

This module is intentionally not a runtime manifold backend. It evaluates one
cell at a time and favors inspectable numerical behavior over throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence


_EPSILON = 1.0e-14
_CONTINUED_FRACTION_EPSILON = 3.0e-14
_CONTINUED_FRACTION_MINIMUM = 1.0e-300
# Narrow but non-degenerate beta distributions can require hundreds of
# Lentz updates. This is a single-cell acceptance oracle, so convergence is
# preferred over a runtime-oriented iteration cap.
_CONTINUED_FRACTION_ITERATIONS = 4096


@dataclass(frozen=True, slots=True)
class BetaState:
    """Admissible beta-FDF moments and their limiting regime."""

    progress: float
    variance: float
    alpha: float | None
    beta: float | None
    regime: str
    clipped: bool


@dataclass(frozen=True, slots=True)
class SingleCellResult:
    """One beta-FDF table evaluation with normalized input moments."""

    state: BetaState
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def _finite(value: float, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def beta_state(
    progress: float,
    variance: float,
    *,
    bounds: str = "clip",
) -> BetaState:
    """Normalize progress moments and identify the beta-distribution limit."""

    if bounds not in {"clip", "error"}:
        raise ValueError("bounds must be clip or error")
    mean = _finite(progress, "progress")
    var = _finite(variance, "variance")
    clipped = False

    if bounds == "error" and not 0.0 <= mean <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    bounded_mean = min(max(mean, 0.0), 1.0)
    clipped = clipped or bounded_mean != mean

    maximum = bounded_mean * (1.0 - bounded_mean)
    if bounds == "error" and not 0.0 <= var <= maximum + _EPSILON:
        raise ValueError("variance must lie in [0, progress*(1-progress)]")
    bounded_variance = min(max(var, 0.0), maximum)
    clipped = clipped or bounded_variance != var

    if bounded_mean <= _EPSILON:
        return BetaState(0.0, 0.0, None, None, "lower_endpoint", clipped)
    if bounded_mean >= 1.0 - _EPSILON:
        return BetaState(1.0, 0.0, None, None, "upper_endpoint", clipped)
    if bounded_variance <= _EPSILON:
        return BetaState(
            bounded_mean,
            0.0,
            None,
            None,
            "delta",
            clipped,
        )
    if maximum - bounded_variance <= _EPSILON:
        return BetaState(
            bounded_mean,
            maximum,
            None,
            None,
            "endpoint_mixture",
            clipped,
        )

    concentration = maximum / bounded_variance - 1.0
    alpha = bounded_mean * concentration
    beta = (1.0 - bounded_mean) * concentration
    return BetaState(
        bounded_mean,
        bounded_variance,
        alpha,
        beta,
        "beta",
        clipped,
    )


def _beta_continued_fraction(alpha: float, beta: float, x: float) -> float:
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CONTINUED_FRACTION_MINIMUM:
        d = _CONTINUED_FRACTION_MINIMUM
    d = 1.0 / d
    value = d

    for iteration in range(1, _CONTINUED_FRACTION_ITERATIONS + 1):
        even = 2 * iteration
        coefficient = (
            iteration
            * (beta - iteration)
            * x
            / ((qam + even) * (alpha + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _CONTINUED_FRACTION_MINIMUM:
            d = _CONTINUED_FRACTION_MINIMUM
        c = 1.0 + coefficient / c
        if abs(c) < _CONTINUED_FRACTION_MINIMUM:
            c = _CONTINUED_FRACTION_MINIMUM
        d = 1.0 / d
        value *= d * c

        coefficient = -(
            (alpha + iteration)
            * (qab + iteration)
            * x
            / ((alpha + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _CONTINUED_FRACTION_MINIMUM:
            d = _CONTINUED_FRACTION_MINIMUM
        c = 1.0 + coefficient / c
        if abs(c) < _CONTINUED_FRACTION_MINIMUM:
            c = _CONTINUED_FRACTION_MINIMUM
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= _CONTINUED_FRACTION_EPSILON:
            return value

    raise RuntimeError("regularized beta continued fraction did not converge")


def _regularized_beta(x: float, alpha: float, beta: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_factor = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(x)
        + beta * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        value = factor * _beta_continued_fraction(alpha, beta, x) / alpha
    else:
        value = 1.0 - (
            factor
            * _beta_continued_fraction(beta, alpha, 1.0 - x)
            / beta
        )
    return min(max(value, 0.0), 1.0)


def _linear_value(grid: tuple[float, ...], values: tuple[float, ...], x: float) -> float:
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
        return values[-1]
    for index in range(1, len(grid)):
        if x <= grid[index]:
            lower = grid[index - 1]
            upper = grid[index]
            position = (x - lower) / (upper - lower)
            return values[index - 1] + position * (
                values[index] - values[index - 1]
            )
    raise RuntimeError("table interval search failed")


def _beta_expectation(
    grid: tuple[float, ...],
    values: tuple[float, ...],
    state: BetaState,
) -> float:
    if state.regime == "lower_endpoint":
        return values[0]
    if state.regime == "upper_endpoint":
        return values[-1]
    if state.regime == "delta":
        return _linear_value(grid, values, state.progress)
    if state.regime == "endpoint_mixture":
        return (
            (1.0 - state.progress) * values[0]
            + state.progress * values[-1]
        )

    assert state.alpha is not None
    assert state.beta is not None
    result = 0.0
    for index in range(1, len(grid)):
        lower = grid[index - 1]
        upper = grid[index]
        probability = _regularized_beta(
            upper, state.alpha, state.beta
        ) - _regularized_beta(lower, state.alpha, state.beta)
        first_moment = state.progress * (
            _regularized_beta(upper, state.alpha + 1.0, state.beta)
            - _regularized_beta(lower, state.alpha + 1.0, state.beta)
        )
        slope = (values[index] - values[index - 1]) / (upper - lower)
        intercept = values[index - 1] - slope * lower
        result += intercept * probability + slope * first_moment
    return result


def evaluate_single_cell(
    *,
    progress: float,
    variance: float,
    grid: Sequence[float],
    outputs: Mapping[str, Sequence[float]],
    bounds: str = "clip",
) -> SingleCellResult:
    """Integrate one piecewise-linear flamelet table against a beta FDF."""

    coordinates = tuple(_finite(value, "grid coordinate") for value in grid)
    if len(coordinates) < 2:
        raise ValueError("grid must contain at least two coordinates")
    if coordinates[0] != 0.0 or coordinates[-1] != 1.0:
        raise ValueError("grid must span the normalized progress domain [0, 1]")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("grid coordinates must be strictly increasing")
    if not outputs:
        raise ValueError("outputs must not be empty")

    state = beta_state(progress, variance, bounds=bounds)
    evaluated: dict[str, float] = {}
    for name, samples in outputs.items():
        logical_name = name.strip()
        if not logical_name:
            raise ValueError("output name must not be empty")
        values = tuple(_finite(value, logical_name) for value in samples)
        if len(values) != len(coordinates):
            raise ValueError(
                f"output {logical_name!r} must have one value per grid coordinate"
            )
        evaluated[logical_name] = _beta_expectation(coordinates, values, state)

    return SingleCellResult(state=state, values=evaluated)
