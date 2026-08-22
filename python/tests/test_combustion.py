from __future__ import annotations

from pathlib import Path
import unittest

import foamnordic as fno


def reaction_rate(
    *, progress: str = "c_tilde", variance: str = "c_var"
) -> fno.Closure:
    return fno.Closure(
        name="reactionRate",
        operator=fno.Operator.model("reaction-rate.fnom"),
        inputs={
            "progress": fno.field(progress),
            "variance": fno.field(variance),
            "temperature": fno.field("T_tilde"),
        },
        outputs={"reaction_rate": fno.field("omega_c")},
    )


def manifold(
    *, progress: str = "c_tilde", variance: str = "c_var"
):
    return fno.Combustion.Manifold.beta_fdf(
        table=Path("flamelet.fnom"),
        progress=fno.field(progress),
        variance=fno.field(variance),
        conditioning={"pressure": fno.field("p")},
        outputs={
            "species": fno.fields("Y_*"),
            "enthalpy": fno.field("h"),
        },
    )


class CombustionDeclarationTests(unittest.TestCase):
    def test_public_api_matches_progress_variable_sketch(self) -> None:
        combustion = fno.Combustion.ProgressVariable(
            reaction_rate=reaction_rate(),
            manifold=manifold(),
        )

        value = combustion.to_plan()
        self.assertEqual(value["contract"], "progress_variable_combustion")
        self.assertEqual(value["reaction_rate"]["inputs"]["progress"]["field"], "c_tilde")
        self.assertEqual(value["manifold"]["kind"], "beta_fdf")
        self.assertEqual(value["manifold"]["integration"], "preintegrated")
        self.assertEqual(
            value["manifold"]["outputs"]["species"],
            {"selection": "fields", "pattern": "Y_*"},
        )
        self.assertEqual(value["coupling"]["source_treatment"], "lagged")
        self.assertIn("Combustion", dir(fno))
        self.assertIn("fields", dir(fno))
        self.assertEqual(
            dir(fno.Combustion),
            ["BetaFDF", "CouplingPolicy", "Manifold", "ProgressVariable"],
        )

    def test_reaction_rate_requires_semantic_ports(self) -> None:
        missing_temperature = fno.Closure(
            name="reactionRate",
            operator=fno.Operator.model("reaction-rate.fnom"),
            inputs={
                "progress": fno.field("c"),
                "variance": fno.field("cVar"),
            },
            outputs={"reaction_rate": fno.field("omega")},
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            fno.Combustion.ProgressVariable(
                reaction_rate=missing_temperature,
                manifold=manifold(progress="c", variance="cVar"),
            )

        wrong_output = fno.Closure(
            name="reactionRate",
            operator=fno.Operator.model("reaction-rate.fnom"),
            inputs=reaction_rate().inputs,
            outputs={"source": fno.field("omega_c")},
        )
        with self.assertRaisesRegex(ValueError, "reaction_rate"):
            fno.Combustion.ProgressVariable(
                reaction_rate=wrong_output,
                manifold=manifold(),
            )

    def test_reaction_rate_and_manifold_moments_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress bindings"):
            fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=manifold(progress="other_c"),
            )
        with self.assertRaisesRegex(ValueError, "variance bindings"):
            fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=manifold(variance="other_var"),
            )

    def test_beta_fdf_contract_is_guarded_and_immutable(self) -> None:
        outputs = {"species": fno.fields("Y_*")}
        declaration = fno.Combustion.Manifold.beta_fdf(
            table="flamelet.fnom",
            progress=fno.field("c"),
            variance=fno.field("cVar"),
            outputs=outputs,
            bounds="error",
        )
        outputs["enthalpy"] = fno.field("h")
        self.assertNotIn("enthalpy", declaration.outputs)
        self.assertEqual(declaration.to_plan()["bounds"], "error")

        with self.assertRaisesRegex(ValueError, ".fnom"):
            fno.Combustion.Manifold.beta_fdf(
                table="flamelet.csv",
                progress=fno.field("c"),
                variance=fno.field("cVar"),
                outputs={"species": fno.fields("Y_*")},
            )
        with self.assertRaisesRegex(ValueError, "preintegrated"):
            fno.Combustion.BetaFDF(
                table="flamelet.fnom",
                progress=fno.field("c"),
                variance=fno.field("cVar"),
                outputs={"species": fno.fields("Y_*")},
                conditioning={},
                integration="python_quadrature",
            )

    def test_field_family_requires_an_explicit_wildcard(self) -> None:
        self.assertEqual(fno.fields("Y_*").pattern, "Y_*")
        with self.assertRaisesRegex(ValueError, "wildcard"):
            fno.fields("Y_CO2")
        with self.assertRaisesRegex(ValueError, "only OpenFOAM"):
            fno.fields("Y_[abc]")

    def test_scaffold_is_guarded_and_contains_no_site_identity(self) -> None:
        root = (
            Path(__file__).resolve().parents[2]
            / "src/foamnordic/template/openfoam/combustion-model"
        )
        files = tuple(path for path in root.iterdir() if path.is_file())
        self.assertGreaterEqual(len(files), 6)
        rendered = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertIn("#error", rendered)
        self.assertIn("@INVOKE_REACTION_RATE_CLOSURE_ONCE@", rendered)
        self.assertIn("@CORRECT_THERMODYNAMICS_ONCE@", rendered)
        self.assertNotIn("project_", rendered)
        self.assertNotIn("/Users/", rendered)
        self.assertNotIn("/scratch/", rendered)

    def test_reaction_rate_adapter_has_a_narrow_equation_contract(self) -> None:
        root = Path(__file__).resolve().parents[2]
        model = root / "src/foamnordic/openfoam/models/reactionRateFjord"
        header = (model / "reactionRateFjord.H").read_text(encoding="utf-8")
        source = (model / "reactionRateFjord.C").read_text(encoding="utf-8")
        registration = (
            root / "src/foamnordic/openfoam/models/makeCombustionModels.C"
        ).read_text(encoding="utf-8")

        self.assertIn("ThermoCombustion<ReactionThermo>", header)
        self.assertIn('TypeName("reactionRateFjord")', header)
        self.assertIn('"progress", "variance", "temperature"', source)
        self.assertIn("closure_->invoke", source)
        self.assertNotIn("thermo().correct", source)
        self.assertIn("dimMass / dimTime", source)
        self.assertIn("dimEnergy / dimVolume / dimTime", source)
        self.assertIn("psiReactionThermo", registration)
        self.assertIn("rhoReactionThermo", registration)

    def test_reaction_rate_dictionary_template_is_concrete(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "src/foamnordic/template/openfoam/combustion-model"
            / "reactionRateFjordProperties.in"
        ).read_text(encoding="utf-8")
        self.assertNotIn("#error", template)
        self.assertIn("reactionRateField", template)
        self.assertIn("reactionRateDimensions", template)
        self.assertIn("@REACTION_RATE_DIMENSIONS@", template)
        self.assertIn("foamNordicClosure", template)


if __name__ == "__main__":
    unittest.main()
