from __future__ import annotations

from contextlib import redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import pydoc
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import foamnordic as fno
from foamnordic.execution.case import (
    PreparedProgram,
    _expression_layout,
    _mesh_commands,
    _package_function,
    _prepare_decomposition,
    _observation_block,
    _output_width,
    _scheme_commands,
    _scheme_requirements,
    _socket_path,
    prepare_case,
    render_dictionary,
    render_transform_dictionary,
    validate_case,
)
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.launch import (
    _host_command,
    _host_group_command,
    _multi_node_host_command,
    _node_ready_path,
)


def write_mesh(case: Path) -> None:
    mesh = case / "constant/polyMesh"
    mesh.mkdir(parents=True, exist_ok=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        (mesh / name).write_text("fixture\n", encoding="utf-8")


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
        interval=100,
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
            openfoam=fno.Slurm.openfoam(nodes=1, ntasks=2),
        ),
    )


class PlanTests(unittest.TestCase):
    def test_local_model_cpu_budget_is_detected_automatically(self) -> None:
        example = example_longship()
        local = fno.Longship(
            case=example.case,
            closures=example.closures,
        )
        with patch(
            "foamnordic.core.native_plan.local_cpu_budget",
            return_value=7,
        ):
            value = local.compile().as_dict()
        self.assertEqual(value["placement"]["closure_cpus_per_node"], "auto")
        self.assertEqual(value["runtime"]["host_cpus_per_task"], 7)

    def test_local_model_cpu_budget_can_be_overridden(self) -> None:
        example = example_longship()
        local = fno.Longship(
            case=example.case,
            closures=example.closures,
            placement=fno.Attached(closure_cpus_per_node=3),
        )
        self.assertEqual(
            local.compile().as_dict()["runtime"]["host_cpus_per_task"],
            3,
        )

    def test_missing_mesh_has_actionable_initialization_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = fno.OpenFOAM.Case(case_dir=root / "case", run_dir=root / "runs")
            with self.assertRaisesRegex(FileNotFoundError, "initialize.*blockMesh"):
                _mesh_commands(fno.Longship(case=case), root / "case")

    def test_block_mesh_initialization_precedes_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_root = root / "case"
            (case_root / "system").mkdir(parents=True)
            (case_root / "system/blockMeshDict").write_text(
                "fixture\n", encoding="utf-8"
            )
            case = fno.OpenFOAM.Case(
                case_dir=case_root,
                run_dir=root / "runs",
            ).initialize(mesh="blockMesh", validate_mesh=True)

            commands = _mesh_commands(fno.Longship(case=case), case_root)

            self.assertEqual(len(commands), 2)
            self.assertTrue(commands[0].startswith("blockMesh -case"))
            self.assertTrue(commands[1].startswith("checkMesh -case"))

    def test_mesh_preparation_reports_compact_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "case"
            for relative in (
                "0/U",
                "system/blockMeshDict",
                "system/controlDict",
                "system/fvSchemes",
                "system/fvSolution",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            case = fno.OpenFOAM.Case(
                name="pitzDaily",
                case_dir=source,
                run_dir=root / "runs",
            ).initialize(mesh="blockMesh", validate_mesh=True)
            longship = fno.Longship(case=case)
            output = io.StringIO()

            with (
                patch("foamnordic.execution.case.subprocess.run", return_value=Mock(returncode=0)),
                redirect_stdout(output),
            ):
                prepare_case(longship, longship.compile(), verbose=True)

            self.assertEqual(
                output.getvalue().splitlines(),
                [
                    "[FoamNordic] Preparing mesh with blockMesh: pitzDaily",
                    "[FoamNordic] Mesh is ready: pitzDaily",
                ],
            )

    def _parallel_case(self, root: Path, ranks: int = 2) -> fno.Longship:
        source = root / "source"
        for relative in (
            "0/U",
            "0/p",
            "constant/turbulenceProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
        ):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        (source / "system/decomposeParDict").write_text(
            "numberOfSubdomains 8;\n"
            "method hierarchical;\n"
            "hierarchicalCoeffs { n (2 2 2); order xyz; }\n",
            encoding="utf-8",
        )
        write_mesh(source)
        return fno.Longship(
            name="parallel-case",
            case=fno.OpenFOAM.Case(
                case_dir=source,
                run_dir=root / "output",
                ranks=ranks,
            ),
        )

    def test_parallel_preparation_normalizes_incompatible_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            longship = self._parallel_case(root)

            def decompose(*_args, **_kwargs):
                copied = next((root / "output/runs").glob("*/case"))
                (copied / "processor0").mkdir()
                (copied / "processor1").mkdir()
                return Mock(returncode=0)

            with patch("foamnordic.execution.case.subprocess.run", side_effect=decompose) as run:
                _, copied, _ = prepare_case(longship, longship.compile())

            dictionary = (copied / "system/decomposeParDict").read_text()
            self.assertIn("numberOfSubdomains 2", dictionary)
            self.assertIn("method          scotch", dictionary)
            self.assertNotIn("hierarchicalCoeffs", dictionary)
            command = " ".join(run.call_args.args[0])
            self.assertIn("numberOfSubdomains -set 2", command)
            self.assertIn("decomposePar", command)

    def test_parallel_preparation_preserves_compatible_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dictionary = Path(directory) / "decomposeParDict"
            original = (
                "numberOfSubdomains 4;\n"
                "method hierarchical;\n"
                "hierarchicalCoeffs { n (2 2 1); order xyz; }\n"
            )
            dictionary.write_text(original, encoding="utf-8")

            changed = _prepare_decomposition(dictionary, 4)

            self.assertFalse(changed)
            self.assertEqual(dictionary.read_text(encoding="utf-8"), original)

    def test_parallel_preparation_rejects_wrong_processor_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            longship = self._parallel_case(root)

            def incomplete(*_args, **_kwargs):
                copied = next((root / "output/runs").glob("*/case"))
                (copied / "processor0").mkdir()
                return Mock(returncode=0)

            with (
                patch("foamnordic.execution.case.subprocess.run", side_effect=incomplete),
                self.assertRaisesRegex(RuntimeError, "does not match Case.ranks"),
            ):
                prepare_case(longship, longship.compile())

    def test_runtime_socket_path_stays_short_for_long_scratch_workspaces(self) -> None:
        work = Path("/scratch") / ("very-long-project-segment/" * 12) / "run"
        first = _socket_path(work, 0)
        second = _socket_path(work, 1)
        self.assertLess(len(os.fsencode(first)), 104)
        self.assertNotEqual(first, second)

    def test_slurm_uses_native_scheduler_vocabulary(self) -> None:
        parameters = inspect.signature(fno.Slurm).parameters
        self.assertIn("openfoam", parameters)
        self.assertIn("model", parameters)
        self.assertNotIn("ntasks", parameters)
        self.assertNotIn("cpus_per_task", parameters)
        self.assertNotIn("mem_per_cpu", parameters)
        self.assertNotIn("solver_tasks", parameters)
        self.assertNotIn("solver_tasks_per_node", parameters)
        self.assertNotIn("solver_cpus_per_task", parameters)

        openfoam = inspect.signature(fno.Slurm.openfoam).parameters
        self.assertEqual(
            tuple(openfoam),
            ("nodes", "ntasks", "cpus_per_task", "mem_per_cpu"),
        )
        model = inspect.signature(fno.Slurm.model).parameters
        self.assertEqual(tuple(model), ("cpus_per_task", "mem_per_cpu"))
        self.assertNotIn("ranks", openfoam)
        self.assertNotIn(
            "ntasks_per_node",
            fno.Slurm.model(cpus_per_task=2).to_plan(),
        )
        self.assertIn("openfoam", dir(fno.Slurm))
        self.assertIn("model", dir(fno.Slurm))

        help_text = pydoc.plain(pydoc.render_doc(fno.Slurm))
        self.assertIn("openfoam(", help_text)
        self.assertIn("model(", help_text)
        self.assertIn("exactly one ClosureHost task per node", help_text)

    def test_multi_node_attached_plan_uses_one_host_per_node(self) -> None:
        example = example_longship()
        longship = fno.Longship(
            case=fno.OpenFOAM.Case(
                case_dir=example.case.case_dir,
                run_dir=example.case.run_dir,
                of_cmd=example.case.of_cmd,
                ranks=16,
            ),
            closures=example.closures,
            scheduler=fno.Slurm(
                account="project_example",
                partition="small",
                time="00:15:00",
                openfoam=fno.Slurm.openfoam(nodes=2, ntasks=16),
                model=fno.Slurm.model(cpus_per_task=2),
            ),
        )

        runtime = longship.compile().as_dict()["runtime"]

        self.assertEqual(runtime["solver_tasks_per_node"], 8)
        self.assertEqual(runtime["host_tasks"], 2)
        self.assertEqual(runtime["placement"]["host_instances"], 2)
        self.assertEqual(runtime["placement"]["data_path"], "shm")

    def test_multi_node_host_specializes_readiness_per_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "program.ready"
            command, expanded = _multi_node_host_command(
                ("worker", "--ready-file", str(ready)),
                (ready,),
                root,
                2,
            )

            self.assertIn("foamnordic.execution.node_host", command)
            self.assertEqual(
                expanded,
                (_node_ready_path(ready, 0), _node_ready_path(ready, 1)),
            )
            configuration = json.loads(
                (root / ".foamnordic/node-host.json").read_text(encoding="utf-8")
            )
            self.assertEqual(configuration["nodes"], 2)

    def test_longship_verbose_displays_resource_plan_at_declaration(self) -> None:
        longship = example_longship()
        with patch("foamnordic.execution.resources.display_resources") as display:
            verbose = fno.Longship(
                case=longship.case,
                closures=longship.closures,
                placement=longship.placement,
                scheduler=longship.scheduler,
                verbose=True,
            )
        display.assert_called_once_with(verbose)

    def test_observe_uses_solver_friendly_cadence(self) -> None:
        parameters = inspect.signature(fno.Observe).parameters
        self.assertIn("interval", parameters)
        self.assertNotIn("every", parameters)
        self.assertNotIn("retention", parameters)
        self.assertNotIn("Retention", dir(fno))

    def test_observe_rejects_missing_field_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "case"
            for relative in (
                "0/U",
                "0/p",
                "system/controlDict",
                "system/fvSchemes",
                "system/fvSolution",
                "constant/transportProperties",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            write_mesh(source)
            case = fno.OpenFOAM.Case(case_dir=source, run_dir=root / "runs")
            transform = fno.Transform(
                name="perturbVelocity",
                operator=fno.Operator.function(lambda velocity: velocity),
                inputs={"velocity": fno.Field("U")},
                outputs={"velocity": fno.Field("U")},
            )
            longship = fno.Longship(
                case=case,
                transforms=(transform,),
                observations=(
                    fno.Observe(
                        summaries={"U": ("l2",), "p": ("mean",), "nut": ("max",)},
                        interval=10,
                    ),
                ),
            )
            fields = {
                "U": Mock(field_class="volVectorField"),
                "p": Mock(field_class="volScalarField"),
            }

            with (
                patch("foamnordic.openfoam.read_case_fields", return_value=(fields, ())),
                self.assertRaisesRegex(
                    ValueError,
                    r"unavailable OpenFOAM field\(s\): nut.*Available.*U, p",
                ),
            ):
                validate_case(longship)

    def test_observe_accepts_registered_adapter_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "case"
            for relative in (
                "0/U",
                "0/p",
                "system/controlDict",
                "system/fvSchemes",
                "system/fvSolution",
                "constant/turbulenceProperties",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            write_mesh(source)
            case = fno.OpenFOAM.Case(case_dir=source, run_dir=root / "runs")
            closure = fno.Closure(
                name="nutFjord",
                operator=fno.Operator.function(lambda velocity_grad, delta: delta),
                inputs={
                    "velocity_grad": fno.Field.grad("U"),
                    "delta": fno.Field.delta(),
                },
                outputs={"nut": fno.Field("nut")},
            )
            longship = fno.Longship(
                case=case,
                closures=(closure,),
                observations=(
                    fno.Observe(summaries={"nut": ("min", "max")}, interval=10),
                ),
            )
            fields = {
                "U": Mock(field_class="volVectorField"),
                "p": Mock(field_class="volScalarField"),
            }

            with patch(
                "foamnordic.openfoam.read_case_fields", return_value=(fields, ())
            ):
                validate_case(longship)

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
        self.assertIn("Operator", dir(fno))
        self.assertIn("Transform", dir(fno))
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
        expected._slurm_start_time.return_value = "2026-08-22T15:42:10"
        with patch("foamnordic.execution.launch.launch", return_value=expected):
            stream = io.StringIO()
            with redirect_stdout(stream):
                actual = example_longship().launch(verbose=True)
            self.assertIs(actual, expected)
            self.assertIn("launched with Job ID: 123456", stream.getvalue())
            self.assertIn(
                "Sailing started at: 2026-08-22T15:42:10",
                stream.getvalue(),
            )
            self.assertIn("Sailing in background: cavity-keqn", stream.getvalue())

            stream = io.StringIO()
            with redirect_stdout(stream):
                example_longship().launch(verbose=False)
            self.assertEqual(stream.getvalue(), "")
            self.assertEqual(expected._wait_for_start.call_count, 2)

    def test_pending_launch_reports_slurm_estimated_start(self) -> None:
        expected = Mock()
        def wait_for_start(timeout, pending_callback):
            pending_callback("123456", "2026-08-22T16:10:00")
            return "123456", "pending"

        expected._wait_for_start.side_effect = wait_for_start
        expected._slurm_start_time.return_value = "2026-08-22T16:10:00"
        with patch("foamnordic.execution.launch.launch", return_value=expected):
            stream = io.StringIO()
            with redirect_stdout(stream):
                example_longship().launch(start_timeout=0.1, verbose=True)
        self.assertIn(
            "Sailing submitted with Job ID: 123456 "
            "(est. start: 2026-08-22T16:10:00)",
            stream.getvalue(),
        )
        self.assertIn(
            "Sailing remains pending with Job ID: 123456 "
            "(est. start: 2026-08-22T16:10:00)",
            stream.getvalue(),
        )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_plan_is_deterministic_and_content_addressed(self) -> None:
        first = example_longship().compile()
        second = example_longship().compile()

        self.assertEqual(first.digest, second.digest)
        self.assertTrue(first.digest.startswith("sha256:"))
        self.assertEqual(first.schema_version, 2)
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
                "openfoam": {
                    "nodes": 1,
                    "ntasks": 2,
                    "cpus_per_task": 1,
                    "mem_per_cpu": None,
                },
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
        self.assertEqual(
            value["closures"][0]["key"],
            {"entropy": [42, 0], "path": [], "scope": "global"},
        )
        self.assertEqual(value["transforms"], [])

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

    def test_operator_model_uses_one_fnom_entry_point(self) -> None:
        operator = fno.Operator.model(Path("model.fnom"))
        self.assertEqual(operator.artifact, Path("model.fnom"))
        self.assertEqual(
            operator.to_plan(),
            {"kind": "model", "manifest": "model.fnom"},
        )
        with self.assertRaisesRegex(ValueError, ".fnom"):
            fno.Operator.model("model.onnx")
        closure = fno.Closure(
            name="closure",
            operator=operator,
            inputs={"velocity": fno.field("U")},
            outputs={"viscosity": fno.field("nut")},
        )
        self.assertIs(closure.operator, operator)
        self.assertEqual(closure.artifact, Path("model.fnom"))

    def test_function_operator_is_declarative_and_transform_launchable(self) -> None:
        operator = fno.Operator.function(lambda velocity: velocity)
        transform = fno.Transform(
            name="perturbVelocity",
            operator=operator,
            inputs={"velocity": fno.field("U")},
            outputs={"velocity": fno.field("U")},
        )
        plan = transform.to_plan()
        self.assertEqual(
            plan["key"],
            {"entropy": [42, 0], "path": [], "scope": "global"},
        )
        self.assertEqual(plan["operator"]["kind"], "function")
        self.assertTrue(plan["operator"]["identity"].startswith("sha256:"))
        self.assertIsNone(operator.artifact)

    def test_function_closure_infers_k_equation_expression_widths(self) -> None:
        case = Mock()
        case.field.side_effect = lambda name: Mock(
            field_class={
                "k": "volScalarField",
                "U": "volVectorField",
            }[name]
        )
        longship = Mock(case=case)

        self.assertEqual(
            _expression_layout(longship, fno.Field("k")).transport_width,
            1,
        )
        self.assertEqual(
            _expression_layout(longship, fno.Field.grad("U")).transport_width,
            9,
        )
        self.assertEqual(
            _expression_layout(longship, fno.Field.delta()).transport_width,
            1,
        )
        self.assertEqual(
            _expression_layout(
                longship,
                fno.Field.coordinate("z"),
            ).transport_width,
            1,
        )

        closure = fno.Closure(
            name="kEqnFjord",
            operator=fno.Operator.function(lambda k, grad_U, delta: k),
            inputs={
                "k": fno.Field("k"),
                "grad_U": fno.Field.grad("U"),
                "delta": fno.Field.delta(),
            },
            outputs={
                "nut": fno.Field("nut"),
                "kProduction": fno.Field("kProduction"),
                "kDissipationCoeff": fno.Field("kDissipationCoeff"),
            },
        )
        self.assertEqual(closure.name, "kEqnFjord")
        self.assertEqual(tuple(closure.inputs), ("k", "grad_U", "delta"))

    def test_closure_names_remain_case_sensitive(self) -> None:
        closure = fno.Closure(
            name="KEqnFjord",
            operator=fno.Operator.model("model.fnom"),
            inputs={"k": fno.Field("k")},
            outputs={"nut": fno.Field("nut")},
        )
        self.assertEqual(closure.name, "KEqnFjord")
        missing_case = Mock()
        missing_case.field.side_effect = KeyError("missing")
        longship = Mock(case=missing_case)
        exact = fno.Closure(
            name="kEqnFjord",
            operator=fno.Operator.function(lambda k: k),
            inputs={"k": fno.Field("k")},
            outputs={"nut": fno.Field("nut")},
        )
        self.assertEqual(_output_width(longship, exact, fno.Field("nut")), 1)
        with self.assertRaisesRegex(KeyError, "case-sensitive adapter"):
            _output_width(longship, closure, fno.Field("nut"))

    def test_function_closure_packages_derived_input_contract(self) -> None:
        try:
            import cloudpickle
        except ImportError:
            self.skipTest("cloudpickle is not installed")

        case = Mock()
        case.field.side_effect = lambda name: Mock(
            field_class={
                "k": "volScalarField",
                "U": "volVectorField",
            }[name]
        )
        longship = Mock(case=case)
        closure = fno.Closure(
            name="kEqnFjord",
            operator=fno.Operator.function(lambda k, grad_U, delta: k),
            inputs={
                "k": fno.Field("k"),
                "grad_U": fno.Field.grad("U"),
                "delta": fno.Field.delta(),
            },
            outputs={
                "nut": fno.Field("nut"),
                "kProduction": fno.Field("kProduction"),
                "kDissipationCoeff": fno.Field("kDissipationCoeff"),
            },
        )
        native = Mock()
        packaged_payload = None

        def capture_bundle(_manifest, payload, *_args):
            nonlocal packaged_payload
            packaged_payload = Path(payload).read_bytes()

        native.write_model_bundle.side_effect = capture_bundle
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(fno, "_native", native, create=True),
            patch.dict(sys.modules, {"foamnordic._native": native}),
        ):
            artifact = _package_function(longship, closure, Path(directory))
            self.assertIsNotNone(artifact)
            assert artifact is not None
            package = cloudpickle.loads(packaged_payload)

        self.assertEqual(package["input_widths"], (1, 9, 1))
        self.assertEqual(
            package["input_layouts"],
            (
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
                {
                    "kind": "tensor",
                    "physical_shape": [3, 3],
                    "transport_width": 9,
                },
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
            ),
        )
        self.assertEqual(
            package["output_layouts"],
            (
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
                {"kind": "scalar", "physical_shape": [], "transport_width": 1},
            ),
        )
        self.assertEqual(
            package["outputs"],
            ("nut", "kProduction", "kDissipationCoeff"),
        )
        native.write_model_bundle.assert_called_once()

    def test_compatibility_seed_is_normalized_to_a_public_key(self) -> None:
        transform = fno.Transform(
            "compatibilitySeed",
            fno.Operator.model("model.fnom"),
            {"velocity": fno.field("U")},
            {"velocity": fno.field("U")},
            "time_step_start",
            7,
        )
        self.assertEqual(transform.key, fno.Random.key(7))
        with self.assertRaisesRegex(ValueError, "either key or seed"):
            fno.Transform(
                name="ambiguousSeed",
                operator=fno.Operator.model("model.fnom"),
                inputs={"velocity": fno.field("U")},
                outputs={"velocity": fno.field("U")},
                key=fno.Random.key(7),
                seed=8,
            )

    def test_transform_renders_logical_keys_separately_from_fields(self) -> None:
        transform = fno.Transform(
            name="predictVelocity",
            operator=fno.Operator.model("velocity.fnom"),
            inputs={
                "x_coordinate": fno.field("x"),
                "pressure": fno.field("p"),
            },
            outputs={"predicted_velocity": fno.field("U")},
            seed=7,
        )
        rendered = render_transform_dictionary(
            transform,
            "unix:///tmp/transform.sock",
            True,
        )
        self.assertIn("exchangeStage   timeStepStart;", rendered)
        self.assertIn("inputKeys       (x_coordinate pressure);", rendered)
        self.assertIn("inputs          (x p);", rendered)
        self.assertIn("outputKeys      (predicted_velocity);", rendered)
        self.assertIn("outputs         (U);", rendered)
        self.assertEqual(transform.to_plan()["key"]["entropy"], [7, 0])

    def test_transform_declares_all_solver_stages_without_aliasing_them(self) -> None:
        expected = {
            "time_step_start": "timeStepStart",
            "outer_corrector": "outerCorrector",
            "pressure_corrected": "pressureCorrected",
            "time_step_end": "timeStepEnd",
        }
        for public, native in expected.items():
            transform = fno.Transform(
                name="lateTransform",
                operator=fno.Operator.model("model.fnom"),
                inputs={"velocity": fno.field("U")},
                outputs={"velocity": fno.field("U")},
                at=public,
            )
            self.assertEqual(transform.to_plan()["at"], public)
            rendered = render_transform_dictionary(
                transform, "unix:///tmp/transform.sock", True
            )
            self.assertIn(f"exchangeStage   {native};", rendered)
        with self.assertRaisesRegex(ValueError, "transform at must"):
            fno.Transform(
                name="invalidTransform",
                operator=fno.Operator.model("model.fnom"),
                inputs={"velocity": fno.field("U")},
                outputs={"velocity": fno.field("U")},
                at="somewhere",
            )

    def test_stock_solver_rejects_only_inner_iteration_stages(self) -> None:
        example = example_longship()
        for stage in ("outer_corrector", "pressure_corrected"):
            transform = fno.Transform(
                name="innerTransform",
                operator=fno.Operator.model("model.fnom"),
                inputs={"velocity": fno.field("U")},
                outputs={"velocity": fno.field("U")},
                at=stage,
            )
            with self.assertRaisesRegex(NotImplementedError, "solver-native"):
                validate_case(
                    fno.Longship(case=example.case, transforms=(transform,))
                )

    def test_nested_openfoam_operations_render_into_closure_dictionary(self) -> None:
        example = example_longship()
        closure = fno.Closure(
            name="kEqnFjord",
            artifact="model.fnom",
            inputs={
                "convection": fno.Math.div("phi", "U"),
                "diffusion": fno.Math.laplacian("nu", "U"),
                "strain": fno.Math.dev(fno.Math.symm(fno.Math.grad("U"))),
            },
            outputs={"nut": fno.field("nut")},
        )
        _, rendered = render_dictionary(
            fno.Longship(case=example.case, closures=(closure,)),
            closure,
            "unix:///tmp/closure.sock",
            True,
        )
        self.assertIn('"div(phi,U)"', rendered)
        self.assertIn('"laplacian(nu,U)"', rendered)
        self.assertIn('"dev(symm(grad(U)))"', rendered)

    def test_python_artifact_selects_managed_resident_automatically(self) -> None:
        example = example_longship()
        closure = fno.Closure(
            name="joblibClosure",
            artifact="model.fnom",
            inputs={"features": fno.field("U")},
            outputs={"prediction": fno.field("nut")},
        )
        longship = fno.Longship(case=example.case, closures=(closure,))
        metadata = {
            "format": "joblib",
            "inputs": [("features", 3, "float64")],
            "outputs": [("prediction", 1, "float64")],
        }
        with patch("foamnordic.execution.launch._artifact_metadata", return_value=metadata):
            command = _host_command(
                longship,
                PreparedProgram(
                    closure,
                    Path("/tmp/ready"),
                    Path("/tmp/program.sock"),
                    None,
                ),
                model_threads=6,
            )
        rendered = " ".join(command)
        self.assertIn(sys.executable, rendered)
        self.assertIn("foamnordic.execution.resident", rendered)
        self.assertIn(f"--connections {longship.case.ranks}", rendered)
        self.assertIn("--threads 6", rendered)
        key_index = command.index("--key")
        self.assertEqual(
            command[key_index + 1],
            '{"entropy":[42,0],"path":[],"scope":"global"}',
        )

    def test_compiled_artifact_selects_managed_resident_automatically(self) -> None:
        example = example_longship()
        closure = fno.Closure(
            name="compiledClosure",
            artifact="model.fnom",
            inputs={"features": fno.field("U")},
            outputs={"prediction": fno.field("nut")},
        )
        longship = fno.Longship(case=example.case, closures=(closure,))
        metadata = {
            "format": "compiled",
            "inputs": [("features", 3, "float64")],
            "outputs": [("prediction", 1, "float64")],
        }
        with patch("foamnordic.execution.launch._artifact_metadata", return_value=metadata):
            command = _host_command(
                longship,
                PreparedProgram(
                    closure,
                    Path("/tmp/ready"),
                    Path("/tmp/program.sock"),
                    None,
                ),
                model_threads=4,
            )
        rendered = " ".join(command)
        self.assertIn("foamnordic.execution.resident", rendered)
        self.assertIn("--threads 4", rendered)
        self.assertNotIn("--key", command)

    def test_transform_selects_the_same_managed_resident(self) -> None:
        example = example_longship()
        transform = fno.Transform(
            name="predictVelocity",
            operator=fno.Operator.model("model.fnom"),
            inputs={"features": fno.field("U")},
            outputs={"prediction": fno.field("U")},
        )
        longship = fno.Longship(case=example.case, transforms=(transform,))
        metadata = {
            "format": "equinox",
            "inputs": [("features", 3, "float64")],
            "outputs": [("prediction", 3, "float64")],
        }
        with patch("foamnordic.execution.launch._artifact_metadata", return_value=metadata):
            command = _host_command(
                longship,
                PreparedProgram(
                    transform,
                    Path("/tmp/ready"),
                    Path("/tmp/program.sock"),
                    None,
                ),
            )
        rendered = " ".join(command)
        self.assertIn("foamnordic.execution.resident", rendered)
        key_index = command.index("--key")
        self.assertEqual(
            command[key_index + 1],
            '{"entropy":[42,0],"path":[],"scope":"global"}',
        )

    def test_multiple_programs_share_the_model_cpu_budget(self) -> None:
        example = example_longship()
        closure = fno.Closure(
            name="joblibClosure",
            artifact="model.fnom",
            inputs={"features": fno.field("U")},
            outputs={"prediction": fno.field("nut")},
        )
        longship = fno.Longship(case=example.case, closures=(closure,))
        prepared = tuple(
            PreparedProgram(
                closure,
                Path(f"/tmp/ready-{index}"),
                Path(f"/tmp/program-{index}.sock"),
                None,
            )
            for index in range(2)
        )
        budgets = []

        def command(_longship, _prepared, model_threads=1):
            budgets.append(model_threads)
            return ("worker", str(model_threads))

        with tempfile.TemporaryDirectory() as directory, patch(
            "foamnordic.execution.launch._host_command",
            side_effect=command,
        ):
            _host_group_command(longship, prepared, Path(directory), host_cpus=5)
        self.assertEqual(budgets, [3, 2])

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "at least the number"
        ):
            _host_group_command(longship, prepared, Path(directory), host_cpus=1)

    def test_ml_expression_schemes_are_planned_without_overriding_defaults(self) -> None:
        example = example_longship()
        closure = fno.Closure(
            name="nutFjord",
            artifact="model.fnom",
            inputs={
                "convection": fno.Math.div("phi", "U"),
                "diffusion": fno.Math.laplacian("nu", "U"),
                "vorticity": fno.Math.curl("U"),
            },
            outputs={"nut": fno.field("nut")},
        )
        longship = fno.Longship(case=example.case, closures=(closure,))
        self.assertEqual(
            _scheme_requirements(longship),
            (
                ("divSchemes", "div(phi,U)", "Gauss linear"),
                (
                    "laplacianSchemes",
                    "laplacian(nu,U)",
                    "Gauss linear corrected",
                ),
                ("gradSchemes", "curl(U)", "Gauss linear"),
            ),
        )
        commands = "\n".join(_scheme_commands(longship, Path("/runs/case")))
        self.assertIn("divSchemes/div(phi,U)", commands)
        self.assertIn("divSchemes/default", commands)
        self.assertIn("foamnordic_default", commands)
        self.assertNotIn("-entry divSchemes/default -set", commands)

    def test_ntasks_must_divide_evenly_across_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            fno.Slurm(
                account="project",
                partition="small",
                time="00:15:00",
                openfoam=fno.Slurm.openfoam(nodes=2, ntasks=3),
            )

    def test_scheduler_directive_values_reject_whitespace(self) -> None:
        with self.assertRaisesRegex(ValueError, "whitespace"):
            fno.Slurm(
                account="project invalid",
                partition="small",
                time="00:15:00",
                openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
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
        self.assertIn("maxRecords  64;", rendered)

    def test_transform_dictionary_embeds_general_observation(self) -> None:
        longship = example_longship()
        transform = fno.Transform(
            name="perturbVelocity",
            operator=fno.Operator.model("velocity.fnom"),
            inputs={"velocity": fno.field("U")},
            outputs={"velocity": fno.field("U")},
        )
        observed = fno.Longship(
            case=longship.case,
            transforms=(transform,),
            observations=(
                fno.Observe(
                    summaries={"U": ("min", "max")},
                    interval=10,
                ),
            ),
        )
        rendered = render_transform_dictionary(
            transform,
            "unix:///tmp/transform.sock",
            True,
            _observation_block(
                observed,
                Path("/runs/cavity/observations.{rank}.jsonl"),
            ),
        )
        self.assertIn("observation", rendered)
        self.assertIn("every       10;", rendered)
        self.assertIn("fields      (U);", rendered)

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

    def test_custom_template_requires_output_keys_for_renamed_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "closure.in"
            template.write_text(
                "outputs (@FOAMNORDIC_OUTPUT_FIELDS@);\n",
                encoding="utf-8",
            )
            case = fno.openfoam.Case(
                case_dir="source",
                run_dir="workspace",
                integration=fno.openfoam.DictionaryTemplate(
                    source=template,
                    destination="constant/modelProperties",
                ),
            )
            closure = fno.Closure(
                name="mapped",
                artifact="mapped.fnom",
                inputs={"velocity": fno.field("U")},
                outputs={"prediction": fno.field("nut")},
            )
            with self.assertRaisesRegex(ValueError, "OUTPUT_KEYS"):
                render_dictionary(
                    fno.Longship(case=case, closures=(closure,)),
                    closure,
                    "unix:///tmp/mapped.sock",
                    True,
                )

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
                    openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
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

    def test_same_field_may_be_transformed_at_distinct_stages(self) -> None:
        run = example_longship()
        start = fno.Transform(
            name="startVelocity",
            operator=fno.Operator.model("start.fnom"),
            inputs={"U": fno.field("U")},
            outputs={"U": fno.field("U")},
            at="time_step_start",
        )
        end = fno.Transform(
            name="endVelocity",
            operator=fno.Operator.model("end.fnom"),
            inputs={"U": fno.field("U")},
            outputs={"U": fno.field("U")},
            at="time_step_end",
        )
        longship = fno.Longship(case=run.case, transforms=(start, end))
        self.assertEqual(len(longship.transforms), 2)

    def test_same_field_and_stage_remain_ambiguous(self) -> None:
        run = example_longship()
        programs = tuple(
            fno.Transform(
                name=f"velocity{index}",
                operator=fno.Operator.model(f"model{index}.fnom"),
                inputs={"U": fno.field("U")},
                outputs={"U": fno.field("U")},
                at="time_step_start",
            )
            for index in range(2)
        )
        with self.assertRaisesRegex(ValueError, "same solver stage"):
            fno.Longship(case=run.case, transforms=programs)


if __name__ == "__main__":
    unittest.main()
