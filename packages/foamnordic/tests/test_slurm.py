from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.node_host import _node_index, _rewrite_ready_file
from foamnordic.execution.resources import resource_values
from foamnordic.execution.slurm import (
    _solver_mpi_directive,
    write_batch,
    write_submission_wrapper,
)

import foamnordic as fno


def test_solver_mpi_directive_selects_available_pmix(mocker) -> None:
    mocker.patch(
        "foamnordic.execution.slurm.shutil.which", return_value="/usr/bin/srun"
    )
    mocker.patch(
        "foamnordic.execution.slurm.subprocess.run",
        return_value=subprocess.CompletedProcess(
            ["srun", "--mpi=list"],
            0,
            stdout=(
                "MPI plugin types are...\n"
                "\tnone\n"
                "\tpmix\n"
                "\tpmi2\n"
                "specific pmix plugin versions available: pmix_v4\n"
            ),
        ),
    )

    assert _solver_mpi_directive(16) == "  --mpi=pmix \\\n"


def test_solver_mpi_directive_preserves_cluster_default_without_pmix(mocker) -> None:
    mocker.patch(
        "foamnordic.execution.slurm.shutil.which", return_value="/usr/bin/srun"
    )
    run = mocker.patch(
        "foamnordic.execution.slurm.subprocess.run",
        return_value=subprocess.CompletedProcess(
            ["srun", "--mpi=list"],
            0,
            stdout="MPI plugin types are...\n\tnone\n\tpmi2\n",
        ),
    )

    assert _solver_mpi_directive(16) == ""
    assert _solver_mpi_directive(1) == ""
    run.assert_called_once()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_node_host_uses_slurm_process_index(mocker) -> None:
    _mock_patch_1 = mocker.patch.dict(os.environ, {"SLURM_PROCID": "1"}, clear=False)
    try:
        assert _node_index(2) == 1
    finally:
        mocker.stop(_mock_patch_1)


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_node_host_rewrites_each_ready_marker() -> None:
    command = [
        "worker",
        "--ready-file",
        "/tmp/first.ready",
        "--ready-file",
        "/tmp/second.ready",
    ]

    assert _rewrite_ready_file(command, ".node1") == [
        "worker",
        "--ready-file",
        "/tmp/first.ready.node1",
        "--ready-file",
        "/tmp/second.ready.node1",
    ]


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def _longship(*, coupled: bool = False) -> fno.Longship:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="workspace",
        of_cmd="openfoam/2512",
        ranks=2,
    )
    closures = ()
    if coupled:
        closures = (
            fno.Closure(
                name="test-closure",
                artifact="model.fnom",
                inputs={"U": fno.field("U")},
                outputs={"nut": fno.field("nut")},
            ),
        )
    return fno.Longship(
        name="baseline",
        case=case,
        closures=closures,
        scheduler=fno.Slurm(
            account="project_example",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(
                nodes=1,
                ntasks=2,
                cpus_per_task=1,
                mem_per_cpu="1G",
            ),
        ),
    )


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_solver_only_batch_has_no_closure_host() -> None:
    longship = _longship()
    runtime = longship.compile().as_dict()["runtime"]
    with tempfile.TemporaryDirectory() as directory:
        batch = write_batch(
            longship,
            runtime,
            Path(directory),
            None,
            ("bash", "-lc", "pimpleFoam -parallel"),
            None,
            120.0,
            30.0,
        )
        script = batch.read_text(encoding="utf-8")
    assert "foamnordic-longship" not in script
    assert "--host srun" not in script
    assert ": >>" in script
    assert "--ntasks=2" in script
    assert "#SBATCH --account=project_example" in script
    assert "#SBATCH --partition=small" in script
    assert "#SBATCH --ntasks=2" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --mem-per-cpu=1G" in script
    assert "--cpus-per-task=1 \\" in script
    assert "--cpu-bind=none \\" in script
    assert "/logs/Sailing_baseline.out" in script
    assert "/logs/Sailing_baseline.log" in script
    assert "unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE" in script
    assert "#SBATCH --open-mode=append" in script
    assert "███████╗" not in script
    assert batch.parent.name == "slurm"


def test_batch_omits_optional_account_and_partition() -> None:
    longship = fno.Longship(
        name="cluster-defaults",
        case=fno.OpenFOAM.Case(
            case_dir="case",
            run_dir="workspace",
            ranks=1,
        ),
        scheduler=fno.Slurm(
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
        ),
    )

    with tempfile.TemporaryDirectory() as directory:
        script = write_batch(
            longship,
            {"solver_tasks_per_node": 1},
            Path(directory),
            None,
            ("pimpleFoam",),
            None,
            120.0,
            30.0,
        ).read_text(encoding="utf-8")

    assert "#SBATCH --account=" not in script
    assert "#SBATCH --partition=" not in script
    assert "#SBATCH --time=00:15:00" in script
    assert longship.scheduler is not None
    assert longship.scheduler.to_plan()["account"] is None
    assert longship.scheduler.to_plan()["partition"] is None


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_coupled_batch_reserves_native_sidecar_task(mocker) -> None:
    mocker.patch(
        "foamnordic.execution.slurm._solver_mpi_directive",
        return_value="  --mpi=pmix \\\n",
    )
    longship = _longship(coupled=True)
    runtime = longship.compile().as_dict()["runtime"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        batch = write_batch(
            longship,
            runtime,
            root,
            ("closure-host",),
            ("pimpleFoam", "-parallel"),
            (root / ".foamnordic/closure.ready",),
            120.0,
            30.0,
        )
        script = batch.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "--host srun --nodes=1 --ntasks=1" in script
    assert "--solver srun --nodes=1 --ntasks=2" in script
    host_step, solver_step = script.split("--solver srun", 1)
    assert "--mpi=pmix" not in host_step
    assert "--mpi=pmix" in solver_step
    assert "--gres=" not in script


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_explicit_resources_keep_solver_and_model_cpu_shapes_separate() -> None:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="workspace",
        of_cmd="openfoam/2512",
        ranks=16,
    )
    closure = fno.Closure(
        name="nutFjord",
        artifact="model.fnom",
        inputs={"U": fno.field("U")},
        outputs={"nut": fno.field("nut")},
    )
    longship = fno.Longship(
        case=case,
        closures=(closure,),
        scheduler=fno.Slurm(
            account="project_example",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(
                nodes=1,
                ntasks=16,
                cpus_per_task=1,
                mem_per_cpu="2G",
            ),
            model=fno.Slurm.model(cpus_per_task=8, mem_per_cpu="1G"),
        ),
    )
    runtime = longship.compile().as_dict()["runtime"]
    values = resource_values(longship)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        script = write_batch(
            longship,
            runtime,
            root,
            ("closure-host",),
            ("pimpleFoam", "-parallel"),
            (root / ".foamnordic/closure.ready",),
            120.0,
            30.0,
        ).read_text(encoding="utf-8")

    assert values["solver_cpus"] == 16
    assert values["model_cpus"] == 8
    assert values["shm"] == 512 * 1024**2
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=24" in script
    assert values["total_memory"] == int(40.5 * 1024**3)
    assert "#SBATCH --mem=41472M" in script
    assert "--host srun --nodes=1 --ntasks=1" in script
    assert "--cpus-per-task=8 --cpu-bind=none" in script
    assert "--solver srun --nodes=1 --ntasks=16" in script


def test_native_programmes_do_not_reserve_model_resources() -> None:
    def scale(velocity):
        return velocity * 1.01

    transform = fno.Transform(
        name="scale",
        operator=fno.Operator.function(scale),
        inputs={"velocity": fno.Field("U")},
        outputs={"velocity": fno.Field("U")},
    )
    longship = fno.Longship(
        case=fno.OpenFOAM.Case(case_dir="case", run_dir="runs"),
        transforms=(transform,),
        scheduler=fno.Slurm(
            account="project_example",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
        ),
    )

    values = resource_values(longship)

    assert values["programs"] == 0
    assert values["model_cpus"] == 0
    assert values["model_memory"] == 0
    assert values["shm"] == 0
    assert values["host_instances"] == 0


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_model_gres_is_reserved_and_assigned_to_the_host_step() -> None:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="workspace",
        of_cmd="openfoam/2512",
        ranks=1,
    )
    closure = fno.Closure(
        name="nutFjord",
        artifact="model.fnom",
        inputs={"U": fno.field("U")},
        outputs={"nut": fno.field("nut")},
    )
    longship = fno.Longship(
        case=case,
        closures=(closure,),
        scheduler=fno.Slurm(
            account="project_example",
            partition="gputest",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(nodes=1, ntasks=1),
            model=fno.Slurm.model(
                cpus_per_task=1,
                gres_model="gh200",
                gres_count=1,
            ),
        ),
    )
    runtime = longship.compile().as_dict()["runtime"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        script = write_batch(
            longship,
            runtime,
            root,
            ("closure-host",),
            ("pimpleFoam",),
            (root / ".foamnordic/closure.ready",),
            120.0,
            30.0,
        ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:gh200:1" in script
    assert "--host srun --nodes=1 --ntasks=1" in script
    assert "--gres=gpu:gh200:1" in script
    solver = script.split("--solver srun", 1)[1]
    assert "--gres=none" in solver


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_multi_node_batch_waits_for_every_attached_host() -> None:
    case = fno.openfoam.Case(
        case_dir="case",
        run_dir="workspace",
        of_cmd="openfoam/2512",
        ranks=16,
    )
    closure = fno.Closure(
        name="nutFjord",
        artifact="model.fnom",
        inputs={"U": fno.field("U")},
        outputs={"nut": fno.field("nut")},
    )
    longship = fno.Longship(
        case=case,
        closures=(closure,),
        scheduler=fno.Slurm(
            account="project_example",
            partition="small",
            time="00:15:00",
            openfoam=fno.Slurm.openfoam(
                nodes=2,
                ntasks=16,
                cpus_per_task=1,
                mem_per_cpu="2G",
            ),
            model=fno.Slurm.model(cpus_per_task=2, mem_per_cpu="1G"),
        ),
    )
    runtime = longship.compile().as_dict()["runtime"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ready = (
            root / ".foamnordic/closure.ready.node0",
            root / ".foamnordic/closure.ready.node1",
        )
        script = write_batch(
            longship,
            runtime,
            root,
            ("node-host",),
            ("pimpleFoam", "-parallel"),
            ready,
            120.0,
            30.0,
        ).read_text(encoding="utf-8")

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=2" in script
    assert "#SBATCH --cpus-per-task=10" in script
    assert "--ready" in script
    assert "closure.ready.node0" in script
    assert "closure.ready.node1" in script
    assert "--host srun --nodes=2 --ntasks=2" in script
    assert "--solver srun --nodes=2 --ntasks=16" in script
    assert "--ntasks-per-node=8" in script


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_submission_wrapper_owns_scancel() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wrapper = write_submission_wrapper(root, root / "slurm/longship.sbatch")
        script = wrapper.read_text(encoding="utf-8")
    assert wrapper.parent.name == "slurm"
    assert 'scancel "$job_id"' in script
    assert 'read -r -u "$FOAMNORDIC_OWNER_FD"' in script
    assert "FOAMNORDIC_ORPHAN_TIMEOUT:-30" in script
    assert ".foamnordic/job.id" in script
    assert 'SLURM_*|SBATCH_*|SRUN_*) unset "$key"' in script
    assert '"${job_id}.batch" "$job_id"' in script
    assert "COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*" in script
    assert 'while squeue --noheader --job "$job_id"' not in script
    assert "seq 1 300" not in script


def test_owner_pipe_eof_cancels_without_waiting_for_allocation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".foamnordic").mkdir()
        wrapper = write_submission_wrapper(root, root / "slurm/longship.sbatch")
        commands = root / "commands"
        commands.mkdir()
        cancelled = root / "cancelled"
        scripts = {
            "sbatch": "#!/bin/bash\nprintf '12345\\n'\n",
            "sacct": (
                "#!/bin/bash\n"
                'if [[ -f "$FNO_CANCELLED" ]]; then\n'
                "  printf 'CANCELLED|\\n'\n"
                "fi\n"
            ),
            "squeue": "#!/bin/bash\nprintf '12345 running\\n'\n",
            "scancel": '#!/bin/bash\n: > "$FNO_CANCELLED"\n',
        }
        for name, contents in scripts.items():
            path = commands / name
            path.write_text(contents, encoding="utf-8")
            path.chmod(0o750)
        reader, writer = os.pipe()
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{commands}:{environment['PATH']}",
                "FNO_CANCELLED": str(cancelled),
                "FOAMNORDIC_OWNER_FD": str(reader),
                "FOAMNORDIC_ORPHAN_TIMEOUT": "0",
            }
        )
        try:
            process = subprocess.Popen(
                (wrapper,),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                pass_fds=(reader,),
            )
            os.close(reader)
            reader = -1
            os.close(writer)
            writer = -1
            _, stderr = process.communicate(timeout=3)
        finally:
            if reader >= 0:
                os.close(reader)
            if writer >= 0:
                os.close(writer)
        was_cancelled = cancelled.is_file()

    assert process.returncode == 130, stderr
    assert was_cancelled


def test_terminal_batch_step_does_not_wait_for_allocation_epilog() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".foamnordic").mkdir()
        wrapper = write_submission_wrapper(root, root / "slurm/longship.sbatch")
        commands = root / "commands"
        commands.mkdir()
        scripts = {
            "sbatch": "#!/bin/bash\nprintf '12345\\n'\n",
            "sacct": (
                "#!/bin/bash\n"
                'case "$*" in\n'
                "  *12345.batch*) printf 'COMPLETED|\\n' ;;\n"
                "esac\n"
            ),
            # A completing allocation may remain visible during its site
            # epilog. The completed batch step is already authoritative.
            "squeue": "#!/bin/bash\nprintf '12345 completing\\n'\n",
            "scancel": "#!/bin/bash\nexit 0\n",
        }
        for name, contents in scripts.items():
            path = commands / name
            path.write_text(contents, encoding="utf-8")
            path.chmod(0o750)
        environment = dict(os.environ)
        environment["PATH"] = f"{commands}:{environment['PATH']}"

        result = subprocess.run(
            (wrapper,),
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
            env=environment,
        )

    assert result.returncode == 0, result.stderr
