"""Backend-neutral mathematics for FoamNordic field models."""

from __future__ import annotations

import builtins
import math as scalar_math
from typing import Any, Iterable

from .core.expressions import FieldExpression, operation


def _namespace(*values: object):
    """Return the NumPy-like namespace owned by the first array value."""

    for value in values:
        namespace = getattr(value, "__array_namespace__", None)
        if callable(namespace):
            return namespace()

        module = type(value).__module__
        if module.startswith(("jax", "jaxlib")):
            import jax.numpy as jnp

            return jnp
        if module.startswith("numpy"):
            import numpy as np

            return np
    return None


def _unary(name: str, value: Any):
    namespace = _namespace(value)
    if namespace is not None:
        return getattr(namespace, name)(value)
    scalar = {"abs": builtins.abs}.get(name, getattr(scalar_math, name, None))
    if scalar is None:
        raise TypeError(f"Math.{name} requires a NumPy or JAX array")
    return scalar(value)


def _reduction(name: str, value: Any, *, axis=None, keepdims: bool = False):
    namespace = _namespace(value)
    if namespace is None:
        if axis is not None or keepdims:
            raise TypeError(
                f"Math.{name} axis and keepdims require a NumPy or JAX array"
            )
        return getattr(builtins, name)(value)
    return getattr(namespace, name)(value, axis=axis, keepdims=keepdims)


class Math:
    """Numerical operations shared by Python scalars, NumPy, and JAX.

    Methods preserve the namespace of their input arrays, so a function can
    run eagerly with NumPy and be traced by JAX without selecting a backend in
    the model body. OpenFOAM field declarations live exclusively under
    :class:`foamnordic.Field`.
    """

    # Element-wise functions shared by Python scalars, NumPy, and JAX.
    @staticmethod
    def abs(value):
        if isinstance(value, FieldExpression):
            return operation("abs", value)
        return _unary("abs", value)

    @staticmethod
    def sqrt(value):
        if isinstance(value, FieldExpression):
            return operation("sqrt", value)
        return _unary("sqrt", value)

    @staticmethod
    def exp(value):
        return _unary("exp", value)

    @staticmethod
    def expm1(value):
        return _unary("expm1", value)

    @staticmethod
    def log(value):
        return _unary("log", value)

    @staticmethod
    def log1p(value):
        return _unary("log1p", value)

    @staticmethod
    def sin(value):
        return _unary("sin", value)

    @staticmethod
    def cos(value):
        return _unary("cos", value)

    @staticmethod
    def tan(value):
        return _unary("tan", value)

    @staticmethod
    def tanh(value):
        return _unary("tanh", value)

    @staticmethod
    def square(value):
        return value * value

    @staticmethod
    def maximum(left, right):
        if isinstance(left, FieldExpression) or isinstance(right, FieldExpression):
            return operation("maximum", left, right)
        namespace = _namespace(left, right)
        if namespace is None:
            return builtins.max(left, right)
        return namespace.maximum(left, right)

    @staticmethod
    def minimum(left, right):
        if isinstance(left, FieldExpression) or isinstance(
            right, FieldExpression
        ):
            return operation("minimum", left, right)
        namespace = _namespace(left, right)
        if namespace is None:
            return builtins.min(left, right)
        return namespace.minimum(left, right)

    @staticmethod
    def clip(value, minimum=None, maximum=None):
        if isinstance(value, FieldExpression):
            if minimum is None and maximum is None:
                raise ValueError("symbolic Math.clip requires at least one bound")
            result = value
            if minimum is not None:
                result = operation("maximum", result, minimum)
            if maximum is not None:
                result = operation("minimum", result, maximum)
            return result
        namespace = _namespace(value, minimum, maximum)
        if namespace is None:
            result = value
            if minimum is not None:
                result = builtins.max(result, minimum)
            if maximum is not None:
                result = builtins.min(result, maximum)
            return result
        return namespace.clip(value, minimum, maximum)

    @staticmethod
    def where(condition, when_true, when_false):
        if isinstance(condition, FieldExpression):
            return operation("where", condition, when_true, when_false)
        namespace = _namespace(condition, when_true, when_false)
        if namespace is None:
            return when_true if condition else when_false
        return namespace.where(condition, when_true, when_false)

    @staticmethod
    def sum(value, *, axis=None, keepdims: bool = False):
        return _reduction("sum", value, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(value, *, axis=None, keepdims: bool = False):
        namespace = _namespace(value)
        if namespace is None:
            if axis is not None or keepdims:
                raise TypeError(
                    "Math.mean axis and keepdims require a NumPy or JAX array"
                )
            values = tuple(value)
            if not values:
                raise ValueError("Math.mean requires at least one value")
            return builtins.sum(values) / len(values)
        return namespace.mean(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(value, *, axis=None, keepdims: bool = False):
        return _reduction("min", value, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(value, *, axis=None, keepdims: bool = False):
        return _reduction("max", value, axis=axis, keepdims=keepdims)

    @staticmethod
    def reshape(value, shape):
        namespace = _namespace(value)
        return value.reshape(shape) if namespace is None else namespace.reshape(value, shape)

    @staticmethod
    def transpose(value, axes=None):
        namespace = _namespace(value)
        if namespace is None:
            raise TypeError("Math.transpose requires a NumPy or JAX array")
        return namespace.transpose(value, axes=axes)

    @staticmethod
    def stack(values: Iterable[Any], *, axis: int = 0):
        arrays = tuple(values)
        namespace = _namespace(*arrays)
        if namespace is None:
            raise TypeError("Math.stack requires NumPy or JAX arrays")
        return namespace.stack(arrays, axis=axis)

    @staticmethod
    def concatenate(values: Iterable[Any], *, axis: int = 0):
        arrays = tuple(values)
        namespace = _namespace(*arrays)
        if namespace is None:
            raise TypeError("Math.concatenate requires NumPy or JAX arrays")
        return namespace.concatenate(arrays, axis=axis)

    @staticmethod
    def einsum(subscripts: str, *operands):
        if any(isinstance(value, FieldExpression) for value in operands):
            compact = subscripts.replace(" ", "")
            if compact == "...ii->..." and len(operands) == 1:
                return operation("trace", operands[0])
            if compact == "...ij,...ij->..." and len(operands) == 2:
                return operation("ddot", operands[0], operands[1])
            raise TypeError(f"unsupported symbolic einsum pattern: {subscripts}")
        namespace = _namespace(*operands)
        if namespace is None:
            raise TypeError("Math.einsum requires NumPy or JAX arrays")
        return namespace.einsum(subscripts, *operands)

    @staticmethod
    def matmul(left, right):
        """Multiply arrays over their final two dimensions."""

        if isinstance(left, FieldExpression) or isinstance(right, FieldExpression):
            return operation("dot", left, right)
        namespace = _namespace(left, right)
        if namespace is None:
            raise TypeError("Math.matmul requires NumPy or JAX arrays")
        return namespace.matmul(left, right)

    @staticmethod
    def outer(left, right):
        """Return the outer product of per-cell vectors."""

        if isinstance(left, (str, FieldExpression)) or isinstance(
            right, (str, FieldExpression)
        ):
            return operation("outer", left, right)

        if getattr(left, "ndim", None) != 2 or getattr(right, "ndim", None) != 2:
            raise ValueError("Math.outer requires per-cell vector fields")
        if left.shape[0] != right.shape[0]:
            raise ValueError("Math.outer requires matching cell dimensions")
        return left[..., :, None] * right[..., None, :]

    @staticmethod
    def trace(value):
        """Return the trace over the final two tensor axes."""

        if isinstance(value, (str, FieldExpression)):
            return operation("trace", value)

        shape = getattr(value, "shape", ())
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("Math.trace requires square final tensor dimensions")
        namespace = _namespace(value)
        if namespace is None:
            raise TypeError("Math.trace requires a NumPy or JAX array")
        return namespace.trace(value, axis1=-2, axis2=-1)

    @staticmethod
    def mag(value):
        """Return per-cell scalar, vector, or tensor magnitude."""

        if isinstance(value, (str, FieldExpression)):
            return operation("mag", value)

        ndim = getattr(value, "ndim", None)
        if not isinstance(ndim, int) or ndim < 1:
            raise TypeError("Math.mag requires an array shaped (n_cells, ...)")
        if ndim == 1:
            return Math.abs(value)
        return Math.sqrt(Math.sum(value * value, axis=tuple(range(1, ndim))))

    @staticmethod
    def symm(value):
        """Return the symmetric part over the final two tensor axes."""

        if isinstance(value, (str, FieldExpression)):
            return operation("symm", value)

        shape = getattr(value, "shape", ())
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("Math.symm requires square final tensor dimensions")
        return 0.5 * (value + value.swapaxes(-1, -2))

    @staticmethod
    def dev(value):
        """Return the deviatoric part over the final two tensor axes."""

        if isinstance(value, (str, FieldExpression)):
            return operation("dev", value)

        shape = getattr(value, "shape", ())
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("Math.dev requires square final tensor dimensions")
        namespace = _namespace(value)
        if namespace is None:
            raise TypeError("Math.dev requires a NumPy or JAX array")
        dimensions = shape[-1]
        trace = Math.trace(value)
        identity = namespace.eye(dimensions, dtype=value.dtype)
        return value - (trace / dimensions)[..., None, None] * identity

    @staticmethod
    def dot(left, right):
        """Contract the final axis of per-cell vectors or tensors."""

        if isinstance(left, (str, FieldExpression)) or isinstance(
            right, (str, FieldExpression)
        ):
            return operation("dot", left, right)

        left_ndim = getattr(left, "ndim", None)
        right_ndim = getattr(right, "ndim", None)
        if left_ndim not in {2, 3} or right_ndim not in {2, 3}:
            raise ValueError("Math.dot supports per-cell vectors and tensors")
        if left.shape[0] != right.shape[0]:
            raise ValueError("Math.dot requires matching cell dimensions")
        if left_ndim == 2 and right_ndim == 2:
            return Math.sum(left * right, axis=-1)
        if left_ndim == 3 and right_ndim == 2:
            return (left @ right[..., None])[..., 0]
        if left_ndim == 2 and right_ndim == 3:
            return (left[..., None, :] @ right)[..., 0, :]
        return left @ right

    @staticmethod
    def ddot(left, right):
        """Double-contract two per-cell tensor fields."""

        if isinstance(left, (str, FieldExpression)) or isinstance(
            right, (str, FieldExpression)
        ):
            return operation("ddot", left, right)

        if getattr(left, "ndim", None) != 3 or getattr(right, "ndim", None) != 3:
            raise ValueError("Math.ddot requires per-cell tensor fields")
        if left.shape != right.shape:
            raise ValueError("Math.ddot requires matching tensor shapes")
        # Contract directly so NumPy does not materialize a full temporary
        # tensor before reducing it. JAX traces the same backend-neutral form.
        return Math.einsum("...ij,...ij->...", left, right)


__all__ = ["FieldExpression", "Math"]


def __dir__() -> list[str]:
    return sorted(__all__)
