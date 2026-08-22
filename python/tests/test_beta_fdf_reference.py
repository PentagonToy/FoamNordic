from __future__ import annotations

import json
from pathlib import Path
import unittest

from foamnordic.combustion.reference import beta_state, evaluate_single_cell


FIXTURE = (
    Path(__file__).parent
    / "fixtures/combustion/beta_fdf_single_cell.json"
)


class SingleCellBetaFDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_golden_cases_cover_all_distribution_limits(self) -> None:
        regimes: set[str] = set()
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                result = evaluate_single_cell(
                    progress=case["progress"],
                    variance=case["variance"],
                    grid=self.fixture["grid"],
                    outputs=self.fixture["outputs"],
                    bounds="error",
                )
                regimes.add(result.state.regime)
                self.assertEqual(result.state.regime, case["regime"])
                if "alpha" in case:
                    self.assertAlmostEqual(result.state.alpha, case["alpha"], places=13)
                    self.assertAlmostEqual(result.state.beta, case["beta"], places=13)
                for name, expected in case["expected"].items():
                    self.assertAlmostEqual(result.values[name], expected, places=12)
                self.assertAlmostEqual(
                    result.values["Y_fuel"] + result.values["Y_product"],
                    1.0,
                    places=13,
                )
        self.assertEqual(
            regimes,
            {"lower_endpoint", "delta", "beta", "endpoint_mixture", "upper_endpoint"},
        )

    def test_general_beta_preserves_linear_table_moments(self) -> None:
        result = evaluate_single_cell(
            progress=0.35,
            variance=0.02,
            grid=self.fixture["grid"],
            outputs=self.fixture["outputs"],
            bounds="error",
        )
        self.assertEqual(result.state.regime, "beta")
        self.assertAlmostEqual(result.values["Y_product"], 0.35, places=12)
        self.assertAlmostEqual(result.values["Y_fuel"], 0.65, places=12)
        self.assertAlmostEqual(result.values["enthalpy"], 720.0, places=10)

    def test_bounds_policy_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress"):
            beta_state(1.1, 0.0, bounds="error")
        with self.assertRaisesRegex(ValueError, "variance"):
            beta_state(0.5, 0.3, bounds="error")

        state = beta_state(0.5, 0.3, bounds="clip")
        self.assertTrue(state.clipped)
        self.assertEqual(state.regime, "endpoint_mixture")
        self.assertEqual(state.variance, 0.25)

    def test_table_shape_and_finite_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            evaluate_single_cell(
                progress=0.5,
                variance=0.01,
                grid=[0.0, 0.5, 0.5, 1.0],
                outputs={"Y": [0.0, 0.5, 0.5, 1.0]},
            )
        with self.assertRaisesRegex(ValueError, "one value per grid"):
            evaluate_single_cell(
                progress=0.5,
                variance=0.01,
                grid=[0.0, 0.5, 1.0],
                outputs={"Y": [0.0, 1.0]},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_single_cell(
                progress=0.5,
                variance=0.01,
                grid=[0.0, 0.5, 1.0],
                outputs={"Y": [0.0, float("nan"), 1.0]},
            )


if __name__ == "__main__":
    unittest.main()
