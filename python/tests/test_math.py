from __future__ import annotations

import math
import unittest

import foamnordic as fno


class MathTests(unittest.TestCase):
    def test_native_selectors_reuse_field_expression_contract(self) -> None:
        self.assertEqual(fno.Math.field("U"), fno.field("U"))
        self.assertEqual(fno.Math.grad("U"), fno.grad("U"))
        self.assertEqual(fno.Math.filter_width(), fno.filter_width())

    def test_grouped_field_vocabulary_is_compact_and_strict(self) -> None:
        self.assertEqual(fno.Field("k"), fno.field("k"))
        self.assertEqual(fno.Field.field("U"), fno.field("U"))
        self.assertEqual(fno.Field.grad("U").canonical, "grad(U)")
        self.assertEqual(fno.Field.delta().canonical, "delta")
        self.assertEqual(fno.Field.coordinate("x").canonical, "x")
        self.assertEqual(fno.Field.div("phi", "U").canonical, "div(phi,U)")
        with self.assertRaisesRegex(ValueError, "x, y, or z"):
            fno.Field.coordinate("X")

    def test_openfoam_operations_build_nested_expression_trees(self) -> None:
        strain = fno.Math.dev(fno.Math.symm(fno.Math.grad("U")))
        contraction = fno.Math.ddot(fno.Math.grad("U"), strain)

        self.assertEqual(strain.canonical, "dev(symm(grad(U)))")
        self.assertEqual(
            contraction.canonical,
            "ddot(grad(U),dev(symm(grad(U))))",
        )
        self.assertTrue(contraction.derived)
        self.assertEqual(
            contraction.to_plan()["arguments"][0],
            {"operation": "grad", "field": "U"},
        )

    def test_openfoam_flux_and_coefficient_operations_are_unambiguous(self) -> None:
        self.assertEqual(fno.Math.div("U").canonical, "div(U)")
        self.assertEqual(fno.Math.div("phi", "U").canonical, "div(phi,U)")
        self.assertEqual(
            fno.Math.laplacian("nu", "U").canonical,
            "laplacian(nu,U)",
        )
        self.assertEqual(fno.Math.curl("U").canonical, "curl(U)")
        self.assertEqual(
            fno.Math.dot(fno.Math.grad("T"), fno.Math.grad("c")).canonical,
            "dot(grad(T),grad(c))",
        )

    def test_openfoam_operation_arity_is_checked_before_launch(self) -> None:
        with self.assertRaisesRegex(ValueError, "div requires 1 or 2"):
            fno.Math.div()
        with self.assertRaisesRegex(TypeError, "positional"):
            fno.Math.curl(fno.Math.grad("U"), fno.Math.grad("T"))
        with self.assertRaisesRegex(ValueError, "OpenFOAM word"):
            fno.Math.grad("grad(U)")

    def test_scalar_elementwise_operations_do_not_require_numpy(self) -> None:
        self.assertEqual(fno.Math.maximum(-1.0, 0.0), 0.0)
        self.assertEqual(fno.Math.minimum(2.0, 1.0), 1.0)
        self.assertAlmostEqual(fno.Math.sqrt(9.0), 3.0)
        self.assertAlmostEqual(fno.Math.exp(0.0), 1.0)
        self.assertAlmostEqual(fno.Math.tanh(0.0), 0.0)
        self.assertAlmostEqual(fno.Math.log(math.e), 1.0)

    def test_numpy_keqn_tensor_path(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional")

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
        self.assertEqual(production.shape, (2,))

    def test_numpy_matmul(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional")

        left = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        right = np.swapaxes(left, -1, -2)

        np.testing.assert_allclose(
            fno.Math.matmul(left, right),
            np.matmul(left, right),
        )

    def test_jax_namespace_survives_jit(self) -> None:
        try:
            import jax
            import jax.numpy as jnp
            import numpy as np
        except ImportError:
            self.skipTest("JAX is optional")

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

    def test_public_directory_contains_math(self) -> None:
        self.assertIn("Math", dir(fno))
        self.assertIn("Field", dir(fno))


if __name__ == "__main__":
    unittest.main()
