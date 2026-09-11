from __future__ import annotations

import math

import pytest

import foamnordic as fno


def test_grouped_field_vocabulary_is_compact_and_strict() -> None:
    assert fno.Field("k") == fno.field("k")
    assert fno.Field.field("U") == fno.field("U")
    assert fno.Field.grad("U").canonical == "grad(U)"
    assert fno.Field.delta().canonical == "delta"
    assert fno.Field.coordinate("x").canonical == "x"
    assert fno.Field.div("phi", "U").canonical == "div(phi,U)"
    with pytest.raises(ValueError, match="x, y, or z"):
        fno.Field.coordinate("X")


def test_openfoam_operations_build_nested_expression_trees() -> None:
    strain = fno.Math.dev(fno.Math.symm(fno.Field.grad("U")))
    contraction = fno.Math.ddot(fno.Field.grad("U"), strain)

    assert strain.canonical == "dev(symm(grad(U)))"
    assert contraction.canonical == "ddot(grad(U),dev(symm(grad(U))))"
    assert contraction.derived
    assert contraction.to_plan()["arguments"][0] == {"operation": "grad", "field": "U"}


def test_openfoam_scalar_algebra_builds_a_safe_expression_tree() -> None:
    strain = fno.Math.symm(fno.Field.grad("U"))
    strain_rate = fno.Math.sqrt(2.0 * fno.Math.ddot(strain, strain))

    assert (
        strain_rate.canonical == "sqrt(multiply(2,ddot(symm(grad(U)),symm(grad(U)))))"
    )
    assert strain_rate.to_plan()["arguments"][0]["arguments"][0] == {
        "operation": "constant",
        "field": "2",
    }
    with pytest.raises(ValueError, match="finite scalar"):
        _ = strain_rate * float("inf")


def test_openfoam_flux_and_coefficient_operations_are_unambiguous() -> None:
    assert fno.Field.div("U").canonical == "div(U)"
    assert fno.Field.div("phi", "U").canonical == "div(phi,U)"
    assert fno.Field.laplacian("nu", "U").canonical == "laplacian(nu,U)"
    assert fno.Field.curl("U").canonical == "curl(U)"
    assert (
        fno.Math.dot(fno.Field.grad("T"), fno.Field.grad("c")).canonical
        == "dot(grad(T),grad(c))"
    )


def test_openfoam_operation_arity_is_checked_before_launch() -> None:
    with pytest.raises(ValueError, match="div requires 1 or 2"):
        fno.Field.div()
    with pytest.raises(TypeError, match="positional"):
        fno.Field.curl(fno.Field.grad("U"), fno.Field.grad("T"))
    with pytest.raises(ValueError, match="OpenFOAM word"):
        fno.Field.grad("grad(U)")


def test_scalar_elementwise_operations_do_not_require_numpy() -> None:
    assert fno.Math.maximum(-1.0, 0.0) == 0.0
    assert fno.Math.minimum(2.0, 1.0) == 1.0
    assert round(abs(fno.Math.sqrt(9.0) - 3.0), 7) == 0
    assert round(abs(fno.Math.exp(0.0) - 1.0), 7) == 0
    assert round(abs(fno.Math.tanh(0.0) - 0.0), 7) == 0
    assert round(abs(fno.Math.log(math.e) - 1.0), 7) == 0


def test_numpy_keqn_tensor_path() -> None:
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy is optional")

    k = np.array([-1.0, 4.0])
    gradient = np.array(
        [
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
            [[0.5, 0.0, 1.0], [0.0, 1.5, 0.0], [2.0, 0.0, 2.5]],
        ]
    )
    positive = fno.Math.maximum(k, 0.0)
    strain = fno.Math.dev(2.0 * fno.Math.symm(gradient))
    production = fno.Math.ddot(gradient, strain)

    np.testing.assert_allclose(positive, np.array([0.0, 4.0]))
    np.testing.assert_allclose(fno.Math.sqrt(positive), np.array([0.0, 2.0]))
    assert production.shape == (2,)


def test_numpy_matmul() -> None:
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy is optional")

    left = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    right = np.swapaxes(left, -1, -2)

    np.testing.assert_allclose(
        fno.Math.matmul(left, right),
        np.matmul(left, right),
    )


def test_jax_namespace_survives_jit() -> None:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError:
        pytest.skip("JAX is optional")

    @jax.jit
    def closure(value):
        return fno.Math.sqrt(fno.Math.maximum(value, 0.0))

    result = closure(jnp.asarray([-1.0, 4.0]))
    np.testing.assert_allclose(np.asarray(result), np.array([0.0, 2.0]))

    @jax.jit
    def tensor_closure(left, right):
        return fno.Math.matmul(left, right)

    left = jnp.eye(3, dtype=jnp.float32)[None, ...]
    result = tensor_closure(left, left)
    expected = np.eye(3, dtype=np.float32)[None, ...]
    np.testing.assert_allclose(np.asarray(result), expected)


def test_public_directory_contains_math() -> None:
    assert "Math" in dir(fno)
    assert "Field" in dir(fno)
