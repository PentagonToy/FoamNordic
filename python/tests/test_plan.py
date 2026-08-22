from __future__ import annotations

from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import foamnordic as fno
from foamnordic._case import render_dictionary
from foamnordic._native_plan import available as native_available


def example_longship() -> fno.Longship:
    case = fno.openfoam.Case(
        case_dir=Path("/cases/cavity"),
        run_dir=Path("/runs/cavity"),
        of_cmd="openfoam/2512",
        shell="bash",
        ranks=2,
    )
    closure = fno.Closure(
        name="kEqnFjord",
        artifact=Path("/models/kEqnFjord.fnom"),
        inputs={
            "k": fno.field("k"),
            "velocity_grad": fno.grad("U"),
            "filter_width": fno.filter_width(),
        },
        outputs={
            "eddy_viscosity": fno.field("nut"),
            "k_production": fno.field("kProduction"),
            "k_dissipation_coeff": fno.field("kDissipationCoeff"),
        },
    )
    observation = fno.Observe(
        summaries={"nut": ("min", "max"), "U": ("l2",)},
        every=100,
        retention=fno.Retention.latest(2, maximum_bytes=64 * 1024 * 1024),
    )
    return fno.Longship(
        name="cavity-keqn",
        case=case,
        closures=(closure,),
        observations=(observation,),
        placement=fno.Attached(data_path="ucx"),
        scheduler=fno.Slurm(
            account="project_example",
            partition="small",
            time="00:15:00",
            nodes=1,
            ntasks=2,
        ),
    )


class PlanTests(unittest.TestCase):
    def test_slurm_uses_native_scheduler_vocabulary(self) -> None:
        parameters = inspect.signature(fno.Slurm).parameters
        self.assertIn("ntasks", parameters)
        self.assertIn("cpus_per_task", parameters)
        self.assertIn("mem_per_cpu", parameters)
        self.assertNotIn("solver_tasks", parameters)
        self.assertNotIn("solver_tasks_per_node", parameters)
        self.assertNotIn("solver_cpus_per_task", parameters)

    def test_longship_name_inherits_from_case(self) -> None:
        case = fno.openfoam.Case(
            name="NACA4412",
            case_dir="case",
            run_dir="workspace",
        )
        run = fno.Longship(case=case)
        self.assertEqual(run.name, "NACA4412")
        self.assertEqual(run.compile().as_dict()["name"], "NACA4412")
        self.assertEqual(run.compile().as_dict()["case"]["name"], "NACA4412")

    def test_public_help_and_dir_are_discoverable(self) -> None:
        self.assertIn("declarative coupled workload", fno.Longship.__doc__)
        self.assertIn("Longship", dir(fno))
        self.assertIn("Run", dir(fno))
        self.assertIn("non-blocking native Longship", fno.Run.__doc__)
        self.assertIn("openfoam", dir(fno))

    def test_launch_verbose_must_be_boolean_before_preparation(self) -> None:
        with self.assertRaisesRegex(TypeError, "verbose"):
            example_longship().launch(verbose="yes")
        with self.assertRaisesRegex(ValueError, "start_timeout"):
            example_longship().launch(start_timeout=0)

    def test_launch_reports_background_sailing_unless_quiet(self) -> None:
        expected = Mock()
        expected._wait_for_start.return_value = ("123456", "running")
        with patch("foamnordic._launch.launch", return_value=expected):
            stream = io.StringIO()
            with redirect_stdout(stream):
                actual = example_longship().launch()
            self.assertIs(actual, expected)
            self.assertIn("launched with Job ID: 123456", stream.getvalue())
            self.assertIn("Sailing in background: cavity-keqn", stream.getvalue())

            stream = io.StringIO()
            with redirect_stdout(stream):
                example_longship().launch(verbose=False)
            self.assertEqual(stream.getvalue(), "")
            self.assertEqual(expected._wait_for_start.call_count, 2)

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_plan_is_deterministic_and_content_addressed(self) -> None:
        first = example_longship().compile()
        second = example_longship().compile()

        self.assertEqual(first.digest, second.digest)
        self.assertTrue(first.digest.startswith("sha256:"))
        self.assertEqual(first.schema_version, 1)
        self.assertEqual(first.as_dict(), second.as_dict())

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_plan_contains_native_lifecycle_invariants(self) -> None:
        value = example_longship().compile().as_dict()

        self.assertEqual(value["case"]["ranks"], 2)
        self.assertEqual(value["placement"]["data_path"], "ucx")
        self.assertEqual(
            value["scheduler"],
            {
                "kind": "slurm",
                "account": "project_example",
                "partition": "small",
                "time": "00:15:00",
                "nodes": 1,
                "ntasks": 2,
                "cpus_per_task": 1,
                "mem_per_cpu": None,
            },
        )
        self.assertEqual(value["runtime"]["lifecycle"]["host_starts_first"], True)
        self.assertEqual(value["runtime"]["lifecycle"]["fail_together"], True)
        self.assertEqual(value["runtime"]["allocation_cpus_per_node"], 3)
        self.assertEqual(value["runtime"]["placement"]["data_path"], "ucx")
        self.assertEqual(
            value["closures"][0]["inputs"]["velocity_grad"],
            {"operation": "grad", "field": "U"},
        )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_solver_only_plan_reserves_no_closure_host(self) -> None:
        example = example_longship()
        pure = fno.Longship(case=example.case, name="openfoam-baseline")
        value = pure.compile().as_dict()
        self.assertEqual(value["closures"], [])
        self.assertEqual(value["runtime"]["host_tasks"], 0)
        self.assertEqual(value["runtime"]["placement"]["kind"], "none")
        self.assertFalse(value["runtime"]["lifecycle"]["host_starts_first"])

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_write_round_trips_as_json(self) -> None:
        plan = example_longship().compile()
        with tempfile.TemporaryDirectory() as directory:
            path = plan.write(Path(directory) / "plan.json")
            value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value, plan.as_dict())

    def test_declarations_copy_mutable_inputs(self) -> None:
        inputs = {"k": fno.field("k")}
        closure = fno.Closure(
            name="closure",
            artifact="model.fnom",
            inputs=inputs,
            outputs={"nut": fno.field("nut")},
        )
        inputs["later"] = fno.field("U")
        self.assertNotIn("later", closure.inputs)

    def test_ntasks_must_divide_evenly_across_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            fno.Slurm(
                account="project",
                partition="small",
                time="00:15:00",
                nodes=2,
                ntasks=3,
            )

    def test_scheduler_directive_values_reject_whitespace(self) -> None:
        with self.assertRaisesRegex(ValueError, "whitespace"):
            fno.Slurm(
                account="project invalid",
                partition="small",
                time="00:15:00",
                nodes=1,
                ntasks=1,
            )

    def test_case_accepts_explicit_openfoam_shell_command(self) -> None:
        case = fno.openfoam.Case(
            case_dir="case",
            run_dir="runs",
            of_cmd="source /opt/openfoam/etc/bashrc",
            shell="zsh",
        )
        self.assertEqual(
            case._toolchain.command,
            "source /opt/openfoam/etc/bashrc",
        )
        self.assertEqual(case.shell, "zsh")

    def test_openfoam_module_name_is_a_convenience_shorthand(self) -> None:
        case = fno.openfoam.Case(
            case_dir="case",
            run_dir="runs",
            of_cmd="openfoam/2512",
        )
        self.assertEqual(case._toolchain.command, "module load openfoam/2512")
        self.assertEqual(case.shell, "bash")

    def test_case_rejects_unsupported_shell(self) -> None:
        with self.assertRaisesRegex(ValueError, "bash or zsh"):
            fno.openfoam.Case(
                case_dir="case",
                run_dir="runs",
                of_cmd="true",
                shell="fish",
            )

    def test_custom_dictionary_template_stays_inside_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the case"):
            fno.openfoam.DictionaryTemplate(
                source="combustion.in",
                destination="../combustionProperties",
            )

    def test_default_dictionary_embeds_rank_sharded_native_observation(self) -> None:
        longship = example_longship()
        destination, rendered = render_dictionary(
            longship,
            longship.closures[0],
            "unix:///tmp/closure.sock",
            True,
            Path("/runs/cavity/observations.{rank}.jsonl"),
        )
        self.assertEqual(destination, Path("constant/turbulenceProperties"))
        self.assertIn('path        "/runs/cavity/observations.{rank}.jsonl";', rendered)
        self.assertIn("fields      (nut U);", rendered)
        self.assertIn("every       100;", rendered)
        self.assertIn("maxRecords  2;", rendered)

    def test_custom_dictionary_template_renders_combustion_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "combustion.in"
            template.write_text(
                "model @FOAMNORDIC_MODEL@;\n"
                "inputs (@FOAMNORDIC_INPUT_KEYS@);\n"
                "outputs (@FOAMNORDIC_OUTPUT_FIELDS@);\n"
                "table @BETA_FDF_TABLE@;\n",
                encoding="utf-8",
            )
            case = fno.openfoam.Case(
                case_dir="source",
                run_dir="workspace",
                integration=fno.openfoam.DictionaryTemplate(
                    source=template,
                    destination="constant/combustionProperties",
                    variables={"BETA_FDF_TABLE": '"betaFdf.tbl"'},
                ),
            )
            closure = fno.Closure(
                name="reactionRateFjord",
                artifact="reaction-rate.fnom",
                inputs={
                    "c_tilde": fno.field("c_tilde"),
                    "c_var": fno.field("c_var"),
                    "T_tilde": fno.field("T_tilde"),
                },
                outputs={"omega_c": fno.field("omega_c")},
            )
            longship = fno.Longship(case=case, closures=(closure,))
            destination, rendered = render_dictionary(
                longship, closure, "unix:///tmp/reaction.sock", True
            )
            self.assertEqual(destination, Path("constant/combustionProperties"))
            self.assertIn("model reactionRateFjord", rendered)
            self.assertIn("c_tilde", rendered)
            self.assertIn("omega_c", rendered)
            self.assertIn('table "betaFdf.tbl"', rendered)

    def test_case_and_scheduler_rank_mismatch_is_rejected(self) -> None:
        run = example_longship()
        with self.assertRaisesRegex(ValueError, "ranks"):
            fno.Longship(
                case=run.case,
                closures=run.closures,
                scheduler=fno.Slurm(
                    account="project",
                    partition="small",
                    time="00:15:00",
                    nodes=1,
                    ntasks=1,
                ),
            )

    def test_output_must_be_a_mutable_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutable field"):
            fno.Closure(
                name="closure",
                artifact="model.fnom",
                inputs={"U": fno.field("U")},
                outputs={"bad": fno.grad("U")},
            )

    def test_duplicate_output_writers_are_rejected(self) -> None:
        run = example_longship()
        duplicate = fno.Closure(
            name="second",
            artifact="second.fnom",
            inputs={"U": fno.field("U")},
            outputs={"prediction": fno.field("nut")},
        )
        with self.assertRaisesRegex(ValueError, "same output field"):
            fno.Longship(case=run.case, closures=run.closures + (duplicate,))


if __name__ == "__main__":
    unittest.main()
