from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import foamnordic as fno
from foamnordic._case import (
    _dimensions,
    render_combustion_dictionary,
    render_combustion_transport_dictionary,
)
from foamnordic._openfoam_reader import Field


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


def combustion_case(root: Path) -> fno.OpenFOAM.Case:
    source = root / "source"
    run = root / "run"
    source.mkdir()
    run.mkdir()
    case = fno.OpenFOAM.Case(case_dir=source, run_dir=run)
    dimensions = {
        "c_tilde": "[0 0 0 0 0 0 0]",
        "c_var": "[0 0 0 0 0 0 0]",
        "T_tilde": "[0 0 0 1 0 0 0]",
        "p": "[1 -1 -2 0 0 0 0]",
        "omega_c": "[1 -3 -1 0 0 0 0]",
        "Y_CH4": "[0 0 0 0 0 0 0]",
        "Y_O2": "[0 0 0 0 0 0 0]",
        "h": "[0 2 -2 0 0 0 0]",
    }
    for name, value in dimensions.items():
        case._fields[name] = Field(
            name=name,
            field_class="volScalarField",
            dimensions=value,
            internal_value=0.0,
            boundary_names=(),
            path=source / "0" / name,
        )
    object.__setattr__(case, "_fields_loaded", True)
    return case


class CombustionDeclarationTests(unittest.TestCase):
    def test_openfoam_dimension_objects_are_lowered_to_seven_exponents(self) -> None:
        self.assertEqual(
            _dimensions((1, -3, -1, 0, 0, 0, 0)),
            "[1 -3 -1 0 0 0 0]",
        )

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
        self.assertIn("progressField", template)
        self.assertIn("@PROGRESS_FIELD@", template)
        self.assertIn("reactionRateDimensions", template)
        self.assertIn("@REACTION_RATE_DIMENSIONS@", template)
        self.assertIn("foamNordicClosure", template)

    def test_progress_variable_lowers_species_family_to_resident_programs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = combustion_case(Path(directory))
            combustion = fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=manifold(),
            )

            reaction, table = combustion.programs(case)

        self.assertEqual(reaction.name, "reactionRate")
        self.assertEqual(table.name, "progressVariableManifold")
        self.assertEqual(
            tuple(table.inputs),
            ("progress", "variance", "pressure"),
        )
        self.assertEqual(
            tuple(table.outputs),
            ("Y_CH4", "Y_O2", "enthalpy"),
        )
        self.assertEqual(table.outputs["Y_CH4"].field_name, "Y_CH4")
        self.assertEqual(table.outputs["enthalpy"].field_name, "h")

    def test_progress_variable_rejects_an_empty_species_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = combustion_case(Path(directory))
            declaration = fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=fno.Combustion.Manifold.beta_fdf(
                    table="flamelet.fnom",
                    progress=fno.field("c_tilde"),
                    variance=fno.field("c_var"),
                    outputs={"species": fno.fields("Z_*")},
                ),
            )
            with self.assertRaisesRegex(ValueError, "matched no"):
                declaration.programs(case)

    def test_combustion_dictionary_wires_two_sessions_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = combustion_case(Path(directory))
            declaration = fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=manifold(),
            )
            longship = fno.Longship(case=case, combustion=declaration)
            reaction, table = longship.closure_programs

            destination, rendered = render_combustion_dictionary(
                longship,
                reaction,
                table,
                "unix:///tmp/reaction.sock",
                "unix:///tmp/manifold.sock",
                True,
            )

        self.assertEqual(destination, Path("constant/combustionProperties"))
        self.assertIn("combustionModel progressVariableFjord", rendered)
        self.assertIn("progressField         c_tilde", rendered)
        self.assertIn("reactionRateField     omega_c", rendered)
        self.assertIn("reactionRateDimensions [1 -3 -1 0 0 0 0]", rendered)
        self.assertIn('address          "unix:///tmp/reaction.sock"', rendered)
        self.assertIn('address          "unix:///tmp/manifold.sock"', rendered)
        self.assertIn("reaction_rate", rendered)
        self.assertIn("enthalpy", rendered)
        self.assertLess(
            rendered.index("reactionRateClosure"),
            rendered.index("manifoldClosure"),
        )
        self.assertIn("Y_CH4", rendered)
        self.assertIn("Y_O2", rendered)

    def test_reference_solver_transport_binds_declared_variance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = combustion_case(Path(directory))
            declaration = fno.Combustion.ProgressVariable(
                reaction_rate=reaction_rate(),
                manifold=manifold(),
            )
            longship = fno.Longship(case=case, combustion=declaration)
            reaction, _ = longship.closure_programs

            destination, rendered = render_combustion_transport_dictionary(
                longship, reaction
            )

        self.assertEqual(
            destination,
            Path("constant/progressVariableTransportProperties"),
        )
        self.assertIn("varianceField     c_var", rendered)
        self.assertIn("progressSchmidt   1.0", rendered)
        self.assertIn("varianceSchmidt   1.0", rendered)

    def test_progress_variable_native_coordinator_preserves_call_order(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        model = root / "src/foamnordic/openfoam/models/progressVariableFjord"
        source = (model / "progressVariableFjord.C").read_text(encoding="utf-8")
        registration = (
            root / "src/foamnordic/openfoam/models/makeCombustionModels.C"
        ).read_text(encoding="utf-8")

        reaction = source.index("reactionRate_->invoke")
        manifold_call = source.index("manifold_->invoke")
        thermo = source.index("this->thermo().correct()")
        self.assertLess(reaction, manifold_call)
        self.assertLess(manifold_call, thermo)
        self.assertIn('sourceTreatment", "lagged"', source)
        self.assertIn('correctionStage", "outerCorrector"', source)
        self.assertIn(
            "makeCombustionTypes(progressVariableFjord, psiReactionThermo)",
            registration,
        )
        self.assertIn(
            "makeCombustionTypes(progressVariableFjord, rhoReactionThermo)",
            registration,
        )

    def test_equation_source_uses_openfoam_combustion_interface(self) -> None:
        root = Path(__file__).resolve().parents[2]
        helper = (
            root
            / "src/foamnordic/openfoam/combustion/progressVariableSource.C"
        ).read_text(encoding="utf-8")
        model = (
            root
            / "src/foamnordic/openfoam/models/progressVariableFjord"
            / "progressVariableFjord.C"
        ).read_text(encoding="utf-8")
        equation = (
            root
            / "src/foamnordic/template/openfoam/combustion-model"
            / "progressVariableEqn.H.in"
        ).read_text(encoding="utf-8")
        variance = (
            root
            / "src/foamnordic/template/openfoam/combustion-model"
            / "varianceEqn.H.in"
        ).read_text(encoding="utf-8")

        self.assertIn("matrix.ref() -= reactionRate", helper)
        self.assertIn("reactionRate.dimensions() * dimVolume", helper)
        self.assertIn("field.name() == progressField_", model)
        self.assertIn("explicitSource(field, source)", model)
        self.assertIn("combustion->R(@PROGRESS_FIELD@)", equation)
        self.assertLess(
            variance.index("@BOUND_VARIANCE_USING_THE_DECLARED_POLICY@"),
            variance.index("combustion->correct()"),
        )

    def test_progress_moment_must_be_a_direct_field(self) -> None:
        closure = fno.Closure(
            name="reactionRate",
            operator=fno.Operator.model("reaction-rate.fnom"),
            inputs={
                "progress": fno.grad("c_tilde"),
                "variance": fno.field("c_var"),
                "temperature": fno.field("T_tilde"),
            },
            outputs={"reaction_rate": fno.field("omega_c")},
        )
        with self.assertRaisesRegex(ValueError, "progress.*solver-owned field"):
            fno.Combustion.ProgressVariable(
                reaction_rate=closure,
                manifold=manifold(),
            )


if __name__ == "__main__":
    unittest.main()
