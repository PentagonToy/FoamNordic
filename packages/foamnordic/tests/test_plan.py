from __future__ import annotations

import inspect
import io
import json
import os
import pydoc
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.case import (
    PreparedProgram,
    _mesh_commands,
    _socket_path,
    prepare_case,
)
from foamnordic.execution.case_validation import validate_case
from foamnordic.execution.function_artifact import (
    package_function as _package_function,
)
from foamnordic.execution.launch import (
    _host_command,
    _host_group_command,
    _multi_node_host_command,
    _node_ready_path,
)
from foamnordic.execution.openfoam_layouts import (
    expression_layout as _expression_layout,
)
from foamnordic.execution.openfoam_layouts import output_width as _output_width
from foamnordic.execution.openfoam_rendering import (
    _scheme_requirements,
    observation_block,
    render_dictionary,
    render_transform_dictionary,
    scheme_commands,
)
from foamnordic.execution.openfoam_templates import (
    prepare_decomposition as _prepare_decomposition,
)
from foamnordic.execution.resources import resource_values

import foamnordic as fno


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


def test_local_model_cpu_budget_is_detected_automatically(mocker) -> None:
    example = example_longship()
    local = fno.Longship(
        case=example.case,
        closures=example.closures,
    )
    _mock_patch_1 = mocker.patch(
        "foamnordic.core.native_plan.local_cpu_budget",
        return_value=7,
    )
    try:
        value = local.compile().as_dict()
    finally:
        mocker.stop(_mock_patch_1)
    assert value["placement"]["model_cpus_per_node"] == "auto"
    assert value["runtime"]["host_cpus_per_task"] == 7


def test_local_model_cpu_budget_can_be_overridden() -> None:
    example = example_longship()
    local = fno.Longship(
        case=example.case,
        closures=example.closures,
        placement=fno.Attached(model_cpus_per_node=3),
    )
    assert local.compile().as_dict()["runtime"]["host_cpus_per_task"] == 3


def test_missing_mesh_has_actionable_initialization_error() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case = fno.OpenFOAM.Case(case_dir=root / "case", run_dir=root / "runs")
        with pytest.raises(FileNotFoundError, match="initialize.*blockMesh"):
            _mesh_commands(fno.Longship(case=case), root / "case")


def test_validate_case_accepts_nonconventional_initial_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "case"
        for relative in (
            "0/alpha.water",
            "0/p_rgh",
            "constant/transportProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
        ):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        write_mesh(source)

        case = fno.OpenFOAM.Case(case_dir=source, run_dir=root / "runs")
        validate_case(fno.Longship(case=case))


def test_validate_case_canonicalizes_symlinked_case_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "real/case"
        for relative in (
            "0/p_rgh",
            "constant/transportProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
        ):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        write_mesh(source)
        alias = root / "alias"
        alias.symlink_to(root / "real", target_is_directory=True)

        case = fno.OpenFOAM.Case(case_dir=alias / "case", run_dir=root / "runs")
        validate_case(fno.Longship(case=case))


def test_block_mesh_initialization_precedes_validation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case_root = root / "case"
        (case_root / "system").mkdir(parents=True)
        (case_root / "system/blockMeshDict").write_text("fixture\n", encoding="utf-8")
        case = fno.OpenFOAM.Case(
            case_dir=case_root,
            run_dir=root / "runs",
        ).initialize(mesh="blockMesh", validate_mesh=True)

        commands = _mesh_commands(fno.Longship(case=case), case_root)

        assert len(commands) == 2
        assert commands[0].startswith("blockMesh -case")
        assert commands[1].startswith("checkMesh -case")


def test_mesh_preparation_reports_compact_progress(mocker) -> None:
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
        _mock_patch_2 = mocker.patch(
            "foamnordic.execution.case_staging.subprocess.run",
            return_value=mocker.Mock(returncode=0),
        )
        try:
            with (
                redirect_stdout(output),
            ):
                prepare_case(longship, longship.compile(), verbose=True)
        finally:
            mocker.stop(_mock_patch_2)

        assert output.getvalue().splitlines() == [
            "[FoamNordic] Preparing mesh with blockMesh: pitzDaily",
            "[FoamNordic] Mesh is ready: pitzDaily",
        ]


def _parallel_case(root: Path, ranks: int = 2) -> fno.Longship:
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


def test_parallel_preparation_normalizes_incompatible_policy(mocker, capsys) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        longship = _parallel_case(root)

        def decompose(*_args, **_kwargs):
            copied = next((root / "output/runs").glob("*/case"))
            (copied / "processor0").mkdir()
            (copied / "processor1").mkdir()
            return mocker.Mock(returncode=0)

        run = mocker.patch(
            "foamnordic.execution.case_staging.subprocess.run", side_effect=decompose
        )
        try:
            _, copied, _ = prepare_case(longship, longship.compile())
        finally:
            mocker.stop(run)

        dictionary = (copied / "system/decomposeParDict").read_text()
        assert "numberOfSubdomains 2" in dictionary
        assert "method          scotch" in dictionary
        assert "hierarchicalCoeffs" not in dictionary
        assert (
            "[FoamNordic] hierarchical decomposition n=(2 x 2 x 2) does not "
            "match ranks=2; using scotch for the isolated run."
            in capsys.readouterr().out
        )
        command = " ".join(run.call_args.args[0])
        assert "numberOfSubdomains -set 2" in command
        assert "decomposePar" in command


def test_parallel_preparation_preserves_compatible_policy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dictionary = Path(directory) / "decomposeParDict"
        original = (
            "numberOfSubdomains 4;\n"
            "method hierarchical;\n"
            "hierarchicalCoeffs { n (2 2 1); order xyz; }\n"
        )
        dictionary.write_text(original, encoding="utf-8")

        changed = _prepare_decomposition(dictionary, 4)

        assert not changed
        assert dictionary.read_text(encoding="utf-8") == original


def test_parallel_preparation_updates_rank_and_preserves_valid_policy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dictionary = Path(directory) / "decomposeParDict"
        dictionary.write_text(
            "numberOfSubdomains 8;\n"
            "method simple;\n"
            "simpleCoeffs { n (2 2 1); delta 0.001; }\n",
            encoding="utf-8",
        )

        changed = _prepare_decomposition(dictionary, 4)
        updated = dictionary.read_text(encoding="utf-8")

        assert changed
        assert "numberOfSubdomains 4;" in updated
        assert "method simple;" in updated
        assert "simpleCoeffs { n (2 2 1); delta 0.001; }" in updated


def test_parallel_preparation_creates_missing_dictionary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dictionary = Path(directory) / "system/decomposeParDict"

        changed = _prepare_decomposition(dictionary, 6)
        created = dictionary.read_text(encoding="utf-8")

        assert changed
        assert "numberOfSubdomains 6;" in created
        assert "method          scotch;" in created


def test_parallel_preparation_rejects_wrong_processor_count(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        longship = _parallel_case(root)

        def incomplete(*_args, **_kwargs):
            copied = next((root / "output/runs").glob("*/case"))
            (copied / "processor0").mkdir()
            return mocker.Mock(returncode=0)

        _mock_patch_4 = mocker.patch(
            "foamnordic.execution.case_staging.subprocess.run", side_effect=incomplete
        )
        try:
            with (
                pytest.raises(RuntimeError, match="does not match Case.ranks"),
            ):
                prepare_case(longship, longship.compile())
        finally:
            mocker.stop(_mock_patch_4)


def test_runtime_socket_path_stays_short_for_long_scratch_workspaces() -> None:
    work = Path("/scratch") / ("very-long-project-segment/" * 12) / "run"
    first = _socket_path(work, 0)
    second = _socket_path(work, 1)
    assert len(os.fsencode(first)) < 104
    assert first != second


def test_slurm_uses_native_scheduler_vocabulary() -> None:
    parameters = inspect.signature(fno.Slurm).parameters
    assert "openfoam" in parameters
    assert "model" in parameters
    assert "ntasks" not in parameters
    assert "cpus_per_task" not in parameters
    assert "mem_per_cpu" not in parameters
    assert "solver_tasks" not in parameters
    assert "solver_tasks_per_node" not in parameters
    assert "solver_cpus_per_task" not in parameters

    openfoam = inspect.signature(fno.Slurm.openfoam).parameters
    assert tuple(openfoam) == ("nodes", "ntasks", "cpus_per_task", "mem_per_cpu")
    model = inspect.signature(fno.Slurm.model).parameters
    assert tuple(model) == (
        "cpus_per_task",
        "mem_per_cpu",
        "gres_model",
        "gres_count",
    )
    assert "ranks" not in openfoam
    assert "ntasks_per_node" not in fno.Slurm.model(cpus_per_task=2).to_plan()
    assert "openfoam" in dir(fno.Slurm)
    assert "model" in dir(fno.Slurm)


def test_slurm_model_gres_requires_a_complete_pair() -> None:
    with pytest.raises(ValueError, match="declared together"):
        fno.Slurm.model(gres_model="gh200")
    with pytest.raises(ValueError, match="declared together"):
        fno.Slurm.model(gres_count=1)
    with pytest.raises(ValueError, match="gres_model"):
        fno.Slurm.model(gres_model="gpu:gh200", gres_count=1)

    assert fno.Slurm.model(
        gres_model="gh200",
        gres_count=1,
    ).to_plan() == {
        "cpus_per_task": 1,
        "mem_per_cpu": None,
        "gres_model": "gh200",
        "gres_count": 1,
    }

    help_text = pydoc.plain(pydoc.render_doc(fno.Slurm))
    assert "openfoam(" in help_text
    assert "model(" in help_text
    assert "exactly one ModelHost task per node" in help_text


def test_multi_node_attached_plan_uses_one_host_per_node() -> None:
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

    assert runtime["solver_tasks_per_node"] == 8
    assert runtime["host_tasks"] == 2
    assert runtime["placement"]["host_instances"] == 2
    assert runtime["placement"]["data_path"] == "shm"


def test_multi_node_host_specializes_readiness_per_node() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ready = root / "program.ready"
        command, expanded = _multi_node_host_command(
            ("worker", "--ready-file", str(ready)),
            (ready,),
            root,
            2,
        )

        assert "foamnordic.execution.node_host" in command
        assert expanded == (_node_ready_path(ready, 0), _node_ready_path(ready, 1))
        configuration = json.loads(
            (root / ".foamnordic/node-host.json").read_text(encoding="utf-8")
        )
        assert configuration["nodes"] == 2


def test_longship_verbose_displays_resource_plan_at_declaration(mocker) -> None:
    longship = example_longship()
    display = mocker.patch("foamnordic.execution.resources.display_resources")
    try:
        verbose = fno.Longship(
            case=longship.case,
            closures=longship.closures,
            placement=longship.placement,
            scheduler=longship.scheduler,
            verbose=True,
        )
    finally:
        mocker.stop(display)
    display.assert_called_once_with(verbose)


def test_observe_uses_solver_friendly_cadence() -> None:
    parameters = inspect.signature(fno.Observe).parameters
    assert "interval" in parameters
    assert "every" not in parameters
    assert "retention" not in parameters
    assert "Retention" not in dir(fno)


def test_observe_rejects_missing_field_before_launch(mocker) -> None:
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
            "U": mocker.Mock(field_class="volVectorField"),
            "p": mocker.Mock(field_class="volScalarField"),
        }
        _mock_patch_6 = mocker.patch(
            "foamnordic.openfoam.read_case_fields", return_value=(fields, ())
        )
        try:
            with (
                pytest.raises(
                    ValueError,
                    match=r"unavailable OpenFOAM field\(s\): nut.*Available.*U, p",
                ),
            ):
                validate_case(longship)
        finally:
            mocker.stop(_mock_patch_6)


def test_observe_accepts_registered_adapter_output(mocker) -> None:
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
            observations=(fno.Observe(summaries={"nut": ("min", "max")}, interval=10),),
        )
        fields = {
            "U": mocker.Mock(field_class="volVectorField"),
            "p": mocker.Mock(field_class="volScalarField"),
        }
        _mock_patch_7 = mocker.patch(
            "foamnordic.openfoam.read_case_fields", return_value=(fields, ())
        )
        try:
            validate_case(longship)
        finally:
            mocker.stop(_mock_patch_7)


def test_longship_name_inherits_from_case() -> None:
    case = fno.openfoam.Case(
        name="NACA4412",
        case_dir="case",
        run_dir="workspace",
    )
    run = fno.Longship(case=case)
    assert run.name == "NACA4412"
    assert run.compile().as_dict()["name"] == "NACA4412"
    assert run.compile().as_dict()["case"]["name"] == "NACA4412"


def test_public_help_and_dir_are_discoverable() -> None:
    assert "declarative coupled workload" in fno.Longship.__doc__
    assert "Longship" in dir(fno)
    assert "Operator" in dir(fno)
    assert "Transform" in dir(fno)
    assert "Run" not in dir(fno)
    assert "Run" in dir(fno.Runtime)
    assert "non-blocking native Longship" in fno.Run.__doc__
    assert "OpenFOAM" in dir(fno)
    assert "openfoam" not in dir(fno)


def test_launch_verbose_must_be_boolean_before_preparation() -> None:
    with pytest.raises(TypeError, match="verbose"):
        example_longship().launch(verbose="yes")
    with pytest.raises(ValueError, match="start_timeout"):
        example_longship().launch(start_timeout=0)
    with pytest.raises(ValueError, match="orphan_timeout"):
        example_longship().launch(orphan_timeout=-1)
    with pytest.raises(ValueError, match="profile_every"):
        example_longship().launch(profile_every=-1)
    with pytest.raises(TypeError, match="profile_every"):
        example_longship().launch(profile_every=True)
    with pytest.raises(ValueError, match="receive_memory_bytes"):
        example_longship().launch(receive_memory_bytes=0)
    with pytest.raises(TypeError, match="receive_memory_bytes"):
        example_longship().launch(receive_memory_bytes=True)


def test_launch_reports_background_sailing_unless_quiet(mocker) -> None:
    expected = mocker.Mock()
    expected._wait_for_start.return_value = ("123456", "running")
    expected._slurm_start_time.return_value = "2026-08-22T15:42:10"
    _mock_patch_8 = mocker.patch(
        "foamnordic.execution.launch.launch", return_value=expected
    )
    try:
        stream = io.StringIO()
        with redirect_stdout(stream):
            actual = example_longship().launch(verbose=True)
        assert actual is expected
        assert "launched with Job ID: 123456" in stream.getvalue()
        assert "Sailing started at: 2026-08-22T15:42:10" in stream.getvalue()
        assert "Sailing in background: cavity-keqn" in stream.getvalue()

        stream = io.StringIO()
        with redirect_stdout(stream):
            example_longship().launch(verbose=False)
        assert stream.getvalue() == ""
        assert expected._wait_for_start.call_count == 2
    finally:
        mocker.stop(_mock_patch_8)


def test_pending_launch_reports_slurm_estimated_start(mocker) -> None:
    expected = mocker.Mock()

    def wait_for_start(timeout, pending_callback):
        pending_callback("123456", "2026-08-22T16:10:00")
        return "123456", "pending"

    expected._wait_for_start.side_effect = wait_for_start
    expected._slurm_start_time.return_value = "2026-08-22T16:10:00"
    _mock_patch_9 = mocker.patch(
        "foamnordic.execution.launch.launch", return_value=expected
    )
    try:
        stream = io.StringIO()
        with redirect_stdout(stream):
            example_longship().launch(start_timeout=0.1, verbose=True)
    finally:
        mocker.stop(_mock_patch_9)
    assert (
        "Sailing submitted with Job ID: 123456 "
        "(est. start: 2026-08-22T16:10:00)" in stream.getvalue()
    )
    assert (
        "Sailing remains pending with Job ID: 123456 "
        "(est. start: 2026-08-22T16:10:00)" in stream.getvalue()
    )


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_plan_is_deterministic_and_content_addressed() -> None:
    first = example_longship().compile()
    second = example_longship().compile()

    assert first.digest == second.digest
    assert first.digest.startswith("sha256:")
    assert first.as_dict() == second.as_dict()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_plan_contains_native_lifecycle_invariants() -> None:
    value = example_longship().compile().as_dict()

    assert value["case"]["ranks"] == 2
    assert value["placement"]["data_path"] == "ucx"
    assert value["scheduler"] == {
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
    }
    assert value["runtime"]["lifecycle"]["host_starts_first"] == True
    assert value["runtime"]["lifecycle"]["fail_together"] == True
    assert value["runtime"]["allocation_cpus_per_node"] == 3
    assert value["runtime"]["placement"]["data_path"] == "ucx"
    assert value["closures"][0]["inputs"]["velocity_grad"] == {
        "operation": "grad",
        "field": "U",
    }
    assert value["closures"][0]["key"] == {
        "entropy": [42, 0],
        "path": [],
        "scope": "global",
    }
    assert value["transforms"] == []


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_solver_only_plan_reserves_no_model_host() -> None:
    example = example_longship()
    pure = fno.Longship(case=example.case, name="openfoam-baseline")
    value = pure.compile().as_dict()
    assert value["closures"] == []
    assert value["runtime"]["host_tasks"] == 0
    assert value["runtime"]["placement"]["kind"] == "none"
    assert not value["runtime"]["lifecycle"]["host_starts_first"]


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_write_round_trips_as_json() -> None:
    plan = example_longship().compile()
    with tempfile.TemporaryDirectory() as directory:
        path = plan.write(Path(directory) / "plan.json")
        value = json.loads(path.read_text(encoding="utf-8"))
    assert value == plan.as_dict()


def test_declarations_copy_mutable_inputs() -> None:
    inputs = {"k": fno.field("k")}
    closure = fno.Closure(
        name="closure",
        artifact="model.fnom",
        inputs=inputs,
        outputs={"nut": fno.field("nut")},
    )
    inputs["later"] = fno.field("U")
    assert "later" not in closure.inputs


def test_operator_model_uses_one_fnom_entry_point() -> None:
    operator = fno.Operator.model(Path("model.fnom"))
    assert operator.artifact == Path("model.fnom")
    assert operator.to_plan() == {"kind": "model", "manifest": "model.fnom"}
    with pytest.raises(ValueError, match=".fnom"):
        fno.Operator.model("model.onnx")
    closure = fno.Closure(
        name="closure",
        operator=operator,
        inputs={"velocity": fno.field("U")},
        outputs={"viscosity": fno.field("nut")},
    )
    assert closure.operator is operator
    assert closure.artifact == Path("model.fnom")


def test_operator_table_is_explicit() -> None:
    operator = fno.Operator.table(Path("manifold.fnom"))
    assert operator.artifact == Path("manifold.fnom")
    assert operator.to_plan() == {"kind": "table", "manifest": "manifold.fnom"}
    with pytest.raises(ValueError, match=".fnom"):
        fno.Operator.table("manifold.npz")


def test_closure_update_is_immutable_and_ordered() -> None:
    reaction = fno.Closure(
        name="progressRateFjord",
        operator=fno.Operator.model("reaction.fnom"),
        inputs={"progress": fno.Field("c")},
        outputs={"omega_c": fno.Field("omega_c")},
    )
    coupled = reaction.update(
        name="thermochemicalState",
        operator=fno.Operator.table("manifold.fnom"),
        inputs={"progress": fno.Field("c")},
        outputs={"H2": fno.Field("H2"), "O2": fno.Field("O2")},
    )

    assert reaction.updates == ()
    assert len(coupled.updates) == 1
    assert coupled.updates[0].name == "thermochemicalState"
    assert coupled.updates[0].operator.kind == "table"
    longship = fno.Longship(case=example_longship().case, closures=(coupled,))
    assert tuple(longship.closure_programs) == (coupled, coupled.updates[0])
    plan = coupled.to_plan()
    assert plan["name"] == "progressRateFjord"
    assert len(plan["updates"]) == 1
    assert plan["updates"][0]["name"] == "thermochemicalState"


def test_closure_updates_count_as_resident_programs() -> None:
    example = example_longship()
    closure = example.closures[0].update(
        name="thermochemicalState",
        operator=fno.Operator.table("manifold.fnom"),
        inputs={"progress": fno.Field("k")},
        outputs={"H2": fno.Field("H2")},
    )
    longship = fno.Longship(
        name=example.name,
        case=example.case,
        closures=(closure,),
        placement=example.placement,
        scheduler=example.scheduler,
    )

    values = resource_values(longship)
    assert values["programs"] == 2
    assert values["model_cpus"] == 2


def test_closure_update_rejects_duplicate_name_and_writer() -> None:
    example = example_longship()
    closure = example.closures[0]
    duplicate_name = closure.update(
        name=closure.name,
        operator=fno.Operator.table("manifold.fnom"),
        inputs={"progress": fno.Field("k")},
        outputs={"H2": fno.Field("H2")},
    )
    with pytest.raises(ValueError, match="names must be unique"):
        fno.Longship(case=example.case, closures=(duplicate_name,))

    duplicate_writer = closure.update(
        name="thermochemicalState",
        operator=fno.Operator.table("manifold.fnom"),
        inputs={"progress": fno.Field("k")},
        outputs={"nut": fno.Field("nut")},
    )
    with pytest.raises(ValueError, match="same output field"):
        fno.Longship(case=example.case, closures=(duplicate_writer,))


def test_custom_dictionary_renders_closure_update_body() -> None:
    with tempfile.TemporaryDirectory() as directory:
        template = Path(directory) / "combustion.in"
        template.write_text(
            "reaction @FOAMNORDIC_MODEL@\n"
            "{\n@FOAMNORDIC_ADDRESS@\n}\n"
            "state @FOAMNORDIC_UPDATE_0_NAME@\n"
            "{\n@FOAMNORDIC_UPDATE_0_BODY@\n}\n",
            encoding="utf-8",
        )
        case = fno.openfoam.Case(
            case_dir="source",
            run_dir="workspace",
            integration=fno.openfoam.DictionaryTemplate(
                source=template,
                destination="constant/combustionProperties",
            ),
        )
        reaction = fno.Closure(
            name="progressRateFjord",
            operator=fno.Operator.model("reaction.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"omega_c": fno.Field("omega_c")},
        )
        coupled = reaction.update(
            name="thermochemicalState",
            operator=fno.Operator.table("manifold.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"H2": fno.Field("H2")},
        )
        destination, rendered = render_dictionary(
            fno.Longship(case=case, closures=(coupled,)),
            coupled,
            "unix:///tmp/reaction.sock",
            True,
            updates=((coupled.updates[0], "unix:///tmp/manifold.sock"),),
        )

    assert destination == Path("constant/combustionProperties")
    assert "state thermochemicalState" in rendered
    assert 'address          "unix:///tmp/manifold.sock";' in rendered
    assert "H2" in rendered


def test_closure_update_requires_explicit_template_slot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        template = Path(directory) / "combustion.in"
        template.write_text(
            "model @FOAMNORDIC_MODEL@;\naddress @FOAMNORDIC_ADDRESS@;\n",
            encoding="utf-8",
        )
        case = fno.openfoam.Case(
            case_dir="source",
            run_dir="workspace",
            integration=fno.openfoam.DictionaryTemplate(
                source=template,
                destination="constant/combustionProperties",
            ),
        )
        closure = fno.Closure(
            name="progressRateFjord",
            operator=fno.Operator.model("reaction.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"omega_c": fno.Field("omega_c")},
        ).update(
            name="thermochemicalState",
            operator=fno.Operator.table("manifold.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"H2": fno.Field("H2")},
        )

        with pytest.raises(ValueError, match="UPDATE_0_BODY"):
            render_dictionary(
                fno.Longship(case=case, closures=(closure,)),
                closure,
                "unix:///tmp/reaction.sock",
                True,
                updates=((closure.updates[0], "unix:///tmp/manifold.sock"),),
            )


def test_prepare_case_allocates_one_endpoint_per_closure_stage(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        (source / "0").mkdir(parents=True)
        (source / "constant").mkdir()
        (source / "system").mkdir()
        (source / "system/controlDict").write_text("functions {}\n", encoding="utf-8")
        (source / "system/fvSchemes").write_text("ddtSchemes {}\n", encoding="utf-8")
        (source / "system/fvSolution").write_text("solvers {}\n", encoding="utf-8")
        write_mesh(source)
        template = root / "combustion.in"
        template.write_text(
            "reaction\n{\n@FOAMNORDIC_ADDRESS@\n}\n"
            "state\n{\n@FOAMNORDIC_UPDATE_0_BODY@\n}\n",
            encoding="utf-8",
        )
        case = fno.openfoam.Case(
            case_dir=source,
            run_dir=root / "runs",
            integration=fno.openfoam.DictionaryTemplate(
                source=template,
                destination="constant/combustionProperties",
            ),
        )
        closure = fno.Closure(
            name="progressRateFjord",
            operator=fno.Operator.model("reaction.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"omega_c": fno.Field("omega_c")},
        ).update(
            name="thermochemicalState",
            operator=fno.Operator.table("manifold.fnom"),
            inputs={"progress": fno.Field("c")},
            outputs={"H2": fno.Field("H2")},
        )
        longship = fno.Longship(case=case, closures=(closure,))
        preparation = mocker.patch(
            "foamnordic.execution.case_staging.subprocess.run"
        )
        try:
            _, prepared_case, programs = prepare_case(
                longship,
                mocker.Mock(digest="digest", as_dict=dict),
                root / "libfoamnordicOpenFOAM.so",
            )
        finally:
            mocker.stop(preparation)

        assert len(programs) == 2
        assert tuple(item.program.name for item in programs) == (
            "progressRateFjord",
            "thermochemicalState",
        )
        assert programs[0].socket != programs[1].socket
        rendered = (prepared_case / "constant/combustionProperties").read_text(
            encoding="utf-8"
        )
        assert str(programs[0].socket) in rendered
        assert str(programs[1].socket) in rendered
        command = " ".join(preparation.call_args.args[0])
        library_directory = root.resolve()
        assert f"FOAM_USER_LIBBIN={library_directory}" in command
        assert f"PATH={library_directory.parent / 'bin'}:" in command
        assert f"LD_LIBRARY_PATH={library_directory}:" in command


def test_function_operator_is_declarative_and_transform_launchable() -> None:
    operator = fno.Operator.function(lambda velocity: velocity)
    transform = fno.Transform(
        name="perturbVelocity",
        operator=operator,
        inputs={"velocity": fno.field("U")},
        outputs={"velocity": fno.field("U")},
    )
    plan = transform.to_plan()
    assert plan["key"] == {"entropy": [42, 0], "path": [], "scope": "global"}
    assert plan["operator"]["kind"] == "function"
    assert plan["operator"]["identity"].startswith("sha256:")
    assert operator.artifact is None


def test_function_closure_infers_k_equation_expression_widths(mocker) -> None:
    case = mocker.Mock()
    case.field.side_effect = lambda name: mocker.Mock(
        field_class={
            "k": "volScalarField",
            "U": "volVectorField",
        }[name]
    )
    longship = mocker.Mock(case=case)

    assert _expression_layout(longship, fno.Field("k")).transport_width == 1
    assert _expression_layout(longship, fno.Field.grad("U")).transport_width == 9
    assert _expression_layout(longship, fno.Field.delta()).transport_width == 1
    assert _expression_layout(longship, fno.Context.delta_time()).transport_width == 1
    assert (
        _expression_layout(
            longship,
            fno.Math.outer(fno.Field("U"), fno.Field("U")),
        ).transport_width
        == 9
    )
    assert (
        _expression_layout(
            longship,
            fno.Math.trace(fno.Math.outer(fno.Field("U"), fno.Field("U"))),
        ).transport_width
        == 1
    )
    with pytest.raises(ValueError, match="diffusivity expression"):
        _expression_layout(
            longship,
            fno.Field.laplacian(fno.Field("U"), "k"),
        )
    with pytest.raises(ValueError, match=r"trace\(\) function input"):
        _expression_layout(longship, fno.Math.trace(fno.Field("U")))
    assert (
        _expression_layout(
            longship,
            fno.Field("k") * fno.Field("k"),
        ).transport_width
        == 1
    )
    assert (
        _expression_layout(
            longship,
            fno.Field("k") * fno.Field("U"),
        ).transport_width
        == 3
    )
    with pytest.raises(ValueError, match="at least one scalar operand"):
        _expression_layout(
            longship,
            fno.Field("U") * fno.Field("U"),
        )
    assert (
        _expression_layout(
            longship,
            fno.Field.coordinate("z"),
        ).transport_width
        == 1
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
    assert closure.name == "kEqnFjord"
    assert tuple(closure.inputs) == ("k", "grad_U", "delta")


def test_closure_names_remain_case_sensitive(mocker) -> None:
    closure = fno.Closure(
        name="KEqnFjord",
        operator=fno.Operator.model("model.fnom"),
        inputs={"k": fno.Field("k")},
        outputs={"nut": fno.Field("nut")},
    )
    assert closure.name == "KEqnFjord"
    missing_case = mocker.Mock()
    missing_case.field.side_effect = KeyError("missing")
    longship = mocker.Mock(case=missing_case)
    exact = fno.Closure(
        name="kEqnFjord",
        operator=fno.Operator.function(lambda k: k),
        inputs={"k": fno.Field("k")},
        outputs={"nut": fno.Field("nut")},
    )
    assert _output_width(longship, exact, fno.Field("nut")) == 1
    with pytest.raises(KeyError, match="case-sensitive adapter"):
        _output_width(longship, closure, fno.Field("nut"))


def test_function_closure_packages_derived_input_contract(mocker) -> None:
    try:
        import cloudpickle
    except ImportError:
        pytest.skip("cloudpickle is not installed")

    case = mocker.Mock()
    case.field.side_effect = lambda name: mocker.Mock(
        field_class={
            "k": "volScalarField",
            "U": "volVectorField",
        }[name]
    )
    longship = mocker.Mock(case=case)
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
    packaged_payload = None

    def capture_bundle(_manifest, payload, *_args):
        nonlocal packaged_payload
        packaged_payload = Path(payload).read_bytes()

    _mock_patch_11 = mocker.patch(
        "foamnordic.execution.function_artifact._write_bundle",
        side_effect=capture_bundle,
    )
    try:
        with (
            tempfile.TemporaryDirectory() as directory,
        ):
            artifact = _package_function(longship, closure, Path(directory))
            assert artifact is not None
            package = cloudpickle.loads(packaged_payload)
    finally:
        mocker.stop(_mock_patch_11)

    assert package["input_widths"] == (1, 9, 1)
    assert package["input_layouts"] == (
        {"kind": "scalar", "physical_shape": [], "transport_width": 1},
        {
            "kind": "tensor",
            "physical_shape": [3, 3],
            "transport_width": 9,
        },
        {"kind": "scalar", "physical_shape": [], "transport_width": 1},
    )
    assert package["output_layouts"] == (
        {"kind": "scalar", "physical_shape": [], "transport_width": 1},
        {"kind": "scalar", "physical_shape": [], "transport_width": 1},
        {"kind": "scalar", "physical_shape": [], "transport_width": 1},
    )
    assert package["outputs"] == ("nut", "kProduction", "kDissipationCoeff")


def test_program_key_requires_the_public_key_type() -> None:
    transform = fno.Transform(
        "keyedTransform",
        fno.Operator.model("model.fnom"),
        {"velocity": fno.field("U")},
        {"velocity": fno.field("U")},
        "time_step_start",
        fno.Random.key(7),
    )
    assert transform.key == fno.Random.key(7)
    with pytest.raises(TypeError, match="Random.Key"):
        fno.Transform(
            name="integerKey",
            operator=fno.Operator.model("model.fnom"),
            inputs={"velocity": fno.field("U")},
            outputs={"velocity": fno.field("U")},
            key=8,
        )


def test_transform_renders_logical_keys_separately_from_fields() -> None:
    transform = fno.Transform(
        name="predictVelocity",
        operator=fno.Operator.model("velocity.fnom"),
        inputs={
            "x_coordinate": fno.field("x"),
            "pressure": fno.field("p"),
        },
        outputs={"predicted_velocity": fno.field("U")},
        key=fno.Random.key(7),
    )
    rendered = render_transform_dictionary(
        transform,
        "unix:///tmp/transform.sock",
        True,
        profile_every=7,
    )
    assert "exchangeStage   timeStepStart;" in rendered
    assert "profileEvery    7;" in rendered
    assert "inputKeys       (x_coordinate pressure);" in rendered
    assert "inputs          (x p);" in rendered
    assert "outputKeys      (predicted_velocity);" in rendered
    assert "outputs         (U);" in rendered
    assert transform.to_plan()["key"]["entropy"] == [7, 0]


def test_transform_renders_patch_selection_separately_from_fields() -> None:
    transform = fno.Transform(
        name="patchVelocity",
        operator=fno.Operator.model("velocity.fnom"),
        inputs={"velocity": fno.Field.patch("U", "inlet")},
        outputs={"velocity": fno.Field.patch("U", "outlet")},
    )
    rendered = render_transform_dictionary(
        transform,
        "unix:///tmp/transform.sock",
        True,
    )
    assert "inputs          (U);" in rendered
    assert "inputPatches    (inlet);" in rendered
    assert "outputs         (U);" in rendered
    assert "outputPatches   (outlet);" in rendered


def test_transform_declares_all_solver_stages_without_aliasing_them() -> None:
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
        assert transform.to_plan()["at"] == public
        rendered = render_transform_dictionary(
            transform, "unix:///tmp/transform.sock", True
        )
        assert f"exchangeStage   {native};" in rendered
    with pytest.raises(ValueError, match="transform at must"):
        fno.Transform(
            name="invalidTransform",
            operator=fno.Operator.model("model.fnom"),
            inputs={"velocity": fno.field("U")},
            outputs={"velocity": fno.field("U")},
            at="somewhere",
        )


def test_standard_solver_rejects_only_inner_iteration_stages() -> None:
    example = example_longship()
    for stage in ("outer_corrector", "pressure_corrected"):
        transform = fno.Transform(
            name="innerTransform",
            operator=fno.Operator.model("model.fnom"),
            inputs={"velocity": fno.field("U")},
            outputs={"velocity": fno.field("U")},
            at=stage,
        )
        with pytest.raises(NotImplementedError, match="solver-native"):
            validate_case(fno.Longship(case=example.case, transforms=(transform,)))


def test_nested_openfoam_operations_render_into_closure_dictionary() -> None:
    example = example_longship()
    closure = fno.Closure(
        name="kEqnFjord",
        artifact="model.fnom",
        inputs={
            "convection": fno.Field.div("phi", "U"),
            "diffusion": fno.Field.laplacian("nu", "U"),
            "strain": fno.Math.dev(fno.Math.symm(fno.Field.grad("U"))),
        },
        outputs={"nut": fno.field("nut")},
    )
    _, rendered = render_dictionary(
        fno.Longship(case=example.case, closures=(closure,)),
        closure,
        "unix:///tmp/closure.sock",
        True,
    )
    assert '"div(phi,U)"' in rendered
    assert '"laplacian(nu,U)"' in rendered
    assert '"dev(symm(grad(U)))"' in rendered


def test_boundary_patches_are_first_class_closure_metadata() -> None:
    example = example_longship()
    inlet = fno.Field.patch("U", "inlet")
    outlet = fno.Field.patch("U", "outlet")
    closure = fno.Closure(
        name="patchVelocity",
        artifact="patch.fnom",
        inputs={"inlet_velocity": inlet},
        outputs={"outlet_velocity": outlet},
    )
    _, rendered = render_dictionary(
        fno.Longship(case=example.case, closures=(closure,)),
        closure,
        "unix:///tmp/patch.sock",
        True,
    )
    assert inlet.to_plan() == {"operation": "field", "field": "U", "patch": "inlet"}
    assert re.search(r"inputPatches\s*\(\s*inlet\s*\)", rendered)
    assert re.search(r"outputPatches\s*\(\s*outlet\s*\)", rendered)


def test_patch_selection_rejects_derived_operations() -> None:
    with pytest.raises(ValueError, match="only valid for stored fields"):
        fno.FieldExpression("grad", "U", patch_name="inlet")


def test_python_artifact_selects_managed_resident_automatically(mocker) -> None:
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
    _mock_patch_12 = mocker.patch(
        "foamnordic.execution.launch._artifact_metadata", return_value=metadata
    )
    try:
        command = _host_command(
            longship,
            PreparedProgram(
                closure,
                Path("/tmp/ready"),
                Path("/tmp/program.sock"),
                None,
            ),
            model_threads=6,
            receive_memory_bytes=268435456,
        )
    finally:
        mocker.stop(_mock_patch_12)
    rendered = " ".join(command)
    assert sys.executable in rendered
    assert "foamnordic.execution.resident" in rendered
    assert f"--connections {longship.case.ranks}" in rendered
    assert "--threads 6" in rendered
    assert "--receive-bytes 268435456" in rendered
    key_index = command.index("--key")
    assert command[key_index + 1] == '{"entropy":[42,0],"path":[],"scope":"global"}'


def test_compiled_artifact_selects_managed_resident_automatically(mocker) -> None:
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
    _mock_patch_13 = mocker.patch(
        "foamnordic.execution.launch._artifact_metadata", return_value=metadata
    )
    try:
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
    finally:
        mocker.stop(_mock_patch_13)
    rendered = " ".join(command)
    assert "foamnordic.execution.resident" in rendered
    assert "--threads 4" in rendered
    assert "--key" not in command


def test_table_artifact_selects_managed_resident_without_onnx_worker(mocker) -> None:
    example = example_longship()
    closure = fno.Closure(
        name="tableClosure",
        operator=fno.Operator.table("table.fnom"),
        inputs={"progress": fno.field("c")},
        outputs={"prediction": fno.field("nut")},
    )
    longship = fno.Longship(case=example.case, closures=(closure,))
    metadata = {
        "format": "table",
        "inputs": [("progress", 1, "float64")],
        "outputs": [("prediction", 1, "float64")],
    }
    _mock_patch_14 = mocker.patch(
        "foamnordic.execution.launch._artifact_metadata",
        return_value=metadata,
    )
    worker = mocker.patch(
        "foamnordic.execution.launch._worker",
        side_effect=AssertionError("table launch must not require the ONNX worker"),
    )
    try:
        command = _host_command(
            longship,
            PreparedProgram(
                closure,
                Path("/tmp/ready"),
                Path("/tmp/program.sock"),
                None,
            ),
            model_threads=2,
        )
    finally:
        mocker.stop(worker)
        mocker.stop(_mock_patch_14)
    assert command[:3] == (
        sys.executable,
        "-m",
        "foamnordic.execution.resident",
    )
    assert "--threads" in command
    assert command[command.index("--threads") + 1] == "2"
    assert "--program" in command
    assert command[command.index("--program") + 1] == "tableClosure"
    assert "--key" not in command


def test_transform_selects_the_same_managed_resident(mocker) -> None:
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
    _mock_patch_16 = mocker.patch(
        "foamnordic.execution.launch._artifact_metadata", return_value=metadata
    )
    try:
        command = _host_command(
            longship,
            PreparedProgram(
                transform,
                Path("/tmp/ready"),
                Path("/tmp/program.sock"),
                None,
            ),
        )
    finally:
        mocker.stop(_mock_patch_16)
    rendered = " ".join(command)
    assert "foamnordic.execution.resident" in rendered
    key_index = command.index("--key")
    assert command[key_index + 1] == '{"entropy":[42,0],"path":[],"scope":"global"}'


def test_multiple_programs_can_borrow_the_full_model_host_cpu_budget(mocker) -> None:
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

    def command(
        _longship,
        _prepared,
        model_threads=1,
        profile_every=0,
        receive_memory_bytes=None,
    ):
        assert profile_every == 0
        assert receive_memory_bytes is None
        budgets.append(model_threads)
        return ("worker", str(model_threads))

    _mock_patch_17 = mocker.patch(
        "foamnordic.execution.launch._host_command",
        side_effect=command,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host, ready = _host_group_command(
                longship, prepared, root, host_cpus=5
            )
            configuration = json.loads(
                (root / ".foamnordic/host-programs.json").read_text()
            )
    finally:
        mocker.stop(_mock_patch_17)
    assert budgets == [5, 5]
    assert host[1:3] == ("-m", "foamnordic.execution.host_group")
    assert ready == (root / ".foamnordic/programs.ready",)
    assert configuration["capacity_units"] == 5
    assert configuration["programs"] == [["5"], ["5"]]
    assert configuration["connector_plugins"] == []

    budgets.clear()
    _mock_patch_18 = mocker.patch(
        "foamnordic.execution.launch._host_command",
        side_effect=command,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            _host_group_command(longship, prepared, Path(directory), host_cpus=1)
    finally:
        mocker.stop(_mock_patch_18)
    assert budgets == [1, 1]


def test_grouped_onnx_program_loads_the_runtime_connector(mocker) -> None:
    example = example_longship()
    closure = example.closures[0]
    prepared = tuple(
        PreparedProgram(
            closure,
            Path(f"/tmp/ready-{index}"),
            Path(f"/tmp/program-{index}.sock"),
            Path(f"/tmp/model-{index}.fnom"),
        )
        for index in range(2)
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        worker = root / "runtime/bin/foamnordic_model_worker"
        worker.parent.mkdir(parents=True)
        worker.touch()
        plugin = root / "runtime/lib/libfoamnordic_onnx_connector.dylib"
        plugin.parent.mkdir(parents=True)
        plugin.touch()

        def command(_longship, item, model_threads=1, **_options):
            return (
                str(worker),
                f"unix://{item.socket}",
                str(item.artifact),
                "--threads",
                str(model_threads),
            )

        _mock_patch_19 = mocker.patch(
            "foamnordic.execution.launch._host_command", side_effect=command
        )
        _mock_patch_20 = mocker.patch(
            "foamnordic.execution.launch._artifact_metadata",
            return_value={"format": "onnx"},
        )
        try:
            _host_group_command(
                longship=example,
                prepared=prepared,
                work_dir=root,
                host_cpus=2,
            )
            configuration = json.loads(
                (root / ".foamnordic/host-programs.json").read_text()
            )
        finally:
            mocker.stop(_mock_patch_20)
            mocker.stop(_mock_patch_19)

    assert configuration["connector_plugins"] == [str(plugin.resolve())]


def test_ml_expression_schemes_are_planned_without_overriding_defaults() -> None:
    example = example_longship()
    closure = fno.Closure(
        name="nutFjord",
        artifact="model.fnom",
        inputs={
            "convection": fno.Field.div("phi", "U"),
            "diffusion": fno.Field.laplacian("nu", "U"),
            "vorticity": fno.Field.curl("U"),
        },
        outputs={"nut": fno.field("nut")},
    )
    longship = fno.Longship(case=example.case, closures=(closure,))
    assert _scheme_requirements(longship) == (
        ("divSchemes", "div(phi,U)", "Gauss linear"),
        (
            "laplacianSchemes",
            "laplacian(nu,U)",
            "Gauss linear corrected",
        ),
        ("gradSchemes", "curl(U)", "Gauss linear"),
    )
    commands = "\n".join(scheme_commands(longship, Path("/runs/case")))
    assert "divSchemes/div(phi,U)" in commands
    assert "divSchemes/default" in commands
    assert "foamnordic_default" in commands
    assert "-entry divSchemes/default -set" not in commands

    recorded = "\n".join(
        scheme_commands(longship, Path("/runs/case"), Path("/runs/work"))
    )
    assert "numerics-changes.tsv" in recorded

    strict = fno.Longship(case=example.case, closures=(closure,), numerics="strict")
    strict_commands = "\n".join(
        scheme_commands(strict, Path("/runs/case"), Path("/runs/work"))
    )
    assert "strict numerics" in strict_commands
    assert "numerics-changes.tsv" not in strict_commands
    assert "-set 'Gauss linear'" not in strict_commands

    with pytest.raises(ValueError, match="numerics"):
        fno.Longship(case=example.case, numerics="relaxed")


def test_ntasks_must_divide_evenly_across_nodes() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        fno.Slurm(
            account="project",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(nodes=2, ntasks=3),
        )


def test_scheduler_directive_values_reject_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        fno.Slurm(
            account="project invalid",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
        )


def test_case_accepts_explicit_openfoam_shell_command() -> None:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="runs",
        of_cmd="source /opt/openfoam/etc/bashrc",
        shell="zsh",
    )
    assert case._toolchain.command == "source /opt/openfoam/etc/bashrc"
    assert case.shell == "zsh"


def test_openfoam_module_name_is_a_convenience_shorthand() -> None:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="runs",
        of_cmd="openfoam/2512",
    )
    assert case._toolchain.command == "module load openfoam/2512"
    assert case.shell == "bash"


def test_case_rejects_unsupported_shell() -> None:
    with pytest.raises(ValueError, match="bash or zsh"):
        fno.openfoam.Case(
            case_dir="case",
            run_dir="runs",
            of_cmd="true",
            shell="fish",
        )


def test_custom_dictionary_template_stays_inside_case() -> None:
    with pytest.raises(ValueError, match="inside the case"):
        fno.openfoam.DictionaryTemplate(
            source="combustion.in",
            destination="../combustionProperties",
        )


def test_default_dictionary_embeds_rank_sharded_native_observation() -> None:
    longship = example_longship()
    destination, rendered = render_dictionary(
        longship,
        longship.closures[0],
        "unix:///tmp/closure.sock",
        True,
        Path("/runs/cavity/observations.{rank}.jsonl"),
        profile_every=11,
    )
    assert destination == Path("constant/turbulenceProperties")
    assert 'path        "/runs/cavity/observations.{rank}.jsonl";' in rendered
    assert "fields      (nut U);" in rendered
    assert "every       100;" in rendered
    assert "maxRecords  64;" in rendered
    assert "profileEvery     11;" in rendered
    assert "profileProgram   kEqnFjord;" in rendered


def test_transform_dictionary_embeds_general_observation() -> None:
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
        observation_block(
            observed,
            Path("/runs/cavity/observations.{rank}.jsonl"),
        ),
    )
    assert "observation" in rendered
    assert "every       10;" in rendered
    assert "fields      (U);" in rendered


def test_custom_dictionary_template_renders_combustion_contract() -> None:
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
            name="customReactionRate",
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
        assert destination == Path("constant/combustionProperties")
        assert "model customReactionRate" in rendered
        assert "c_tilde" in rendered
        assert "omega_c" in rendered
        assert 'table "betaFdf.tbl"' in rendered


def test_custom_template_requires_output_keys_for_renamed_ports() -> None:
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
        with pytest.raises(ValueError, match="OUTPUT_KEYS"):
            render_dictionary(
                fno.Longship(case=case, closures=(closure,)),
                closure,
                "unix:///tmp/mapped.sock",
                True,
            )


def test_case_and_scheduler_rank_mismatch_is_rejected() -> None:
    run = example_longship()
    with pytest.raises(ValueError, match="ranks"):
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


def test_output_must_be_a_mutable_field() -> None:
    with pytest.raises(ValueError, match="mutable field"):
        fno.Closure(
            name="closure",
            artifact="model.fnom",
            inputs={"U": fno.field("U")},
            outputs={"bad": fno.grad("U")},
        )


def test_duplicate_output_writers_are_rejected() -> None:
    run = example_longship()
    duplicate = fno.Closure(
        name="second",
        artifact="second.fnom",
        inputs={"U": fno.field("U")},
        outputs={"prediction": fno.field("nut")},
    )
    with pytest.raises(ValueError, match="same output field"):
        fno.Longship(case=run.case, closures=run.closures + (duplicate,))


def test_same_field_may_be_transformed_at_distinct_stages() -> None:
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
    assert len(longship.transforms) == 2


def test_same_field_and_stage_remain_ambiguous() -> None:
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
    with pytest.raises(ValueError, match="same solver stage"):
        fno.Longship(case=run.case, transforms=programs)
