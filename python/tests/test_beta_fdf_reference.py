from __future__ import annotations

import json
from pathlib import Path
import unittest

from foamnordic.combustion.reference import (
    beta_state,
    evaluate_lagged_mass_source_trajectory,
    evaluate_single_cell,
)


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

    def test_lagged_source_then_manifold_trajectory(self) -> None:
        density = 1.2

        def reaction_rate(
            progress: float,
            variance: float,
            temperature: float,
        ) -> float:
            self.assertEqual(variance, 0.01)
            self.assertEqual(temperature, 900.0)
            return density * 2.0 * (1.0 - progress)

        trajectory = evaluate_lagged_mass_source_trajectory(
            progress=0.2,
            variance=0.01,
            temperature=900.0,
            density=density,
            delta_t=0.1,
            steps=3,
            reaction_rate=reaction_rate,
            grid=[0.0, 0.5, 1.0],
            outputs={"Y_product": [0.0, 0.5, 1.0]},
        )

        self.assertEqual(len(trajectory), 3)
        expected_progress = (0.36, 0.488, 0.5904)
        expected_sources = (1.92, 1.536, 1.2288)
        for step, progress, source in zip(
            trajectory, expected_progress, expected_sources
        ):
            self.assertAlmostEqual(step.source_used, source, places=13)
            self.assertAlmostEqual(step.progress_after, progress, places=13)
            self.assertAlmostEqual(
                step.manifold["Y_product"], progress, places=12
            )
        self.assertAlmostEqual(
            trajectory[0].source_next,
            trajectory[1].source_used,
            places=13,
        )


if __name__ == "__main__":
    unittest.main()
