from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest
from foamnordic.core.managed import generated_kind, mark_generated
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.host_group import main as run_host_group
from foamnordic.execution.launch import _openfoam_library, _submission_environment
from foamnordic.execution.run import _launch_process, _longship_executable

import foamnordic as fno


def runtime_available() -> bool:
    if not native_available():
        return False
    try:
        _longship_executable()
    except RuntimeError:
        return False
    return True


def write_mesh(case: Path) -> None:
    mesh = case / "constant/polyMesh"
    mesh.mkdir(parents=True, exist_ok=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        (mesh / name).write_text("fixture\n", encoding="utf-8")


def test_multiple_workers_publish_one_group_readiness_marker() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        aggregate = root / "all.ready"
        commands = []
        ready_files = []
        for index in range(2):
            ready = root / f"worker-{index}.ready"
            ready_files.append(str(ready))
            script = (
                "from pathlib import Path; import time; "
                f"Path({str(ready)!r}).touch(); "
                f"aggregate=Path({str(aggregate)!r}); "
                "\nwhile not aggregate.exists(): time.sleep(0.01)"
            )
            commands.append([sys.executable, "-c", script])
        configuration = root / "group.json"
        configuration.write_text(
            json.dumps(
                {
                    "commands": commands,
                    "ready_files": ready_files,
                    "aggregate_ready": str(aggregate),
                }
            ),
            encoding="utf-8",
        )
        assert run_host_group(str(configuration)) == 0
        assert not aggregate.exists()
        assert all(not Path(path).exists() for path in ready_files)


def test_slurm_identity_and_timing_are_finalized_after_completion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        work = root / "case-temporary"
        mark_generated(work, kind="run")
        logs = work / "logs"
        logs.mkdir(parents=True)
        sailing = logs / "Sailing_case.log"
        harbor = logs / "Harbor_case.log"
        solver = logs / "Sailing_case.out"
        job_file = work / ".foamnordic/job.id"
        job_file.parent.mkdir()
        job_file.write_text("783987\n", encoding="utf-8")
        command = (
            "/bin/sh",
            "-c",
            f"printf 'ExecutionTime = 1.25 s  ClockTime = 2 s\\n' > {solver}",
        )
        run = _launch_process(
            command,
            work_dir=work,
            process_log=sailing,
            longship_log=sailing,
            host_log=harbor,
            solver_log=solver,
            name="case_name",
            job_file=job_file,
        )
        result = run.stop(timeout=3)

        assert result.work_dir.name == "case-name-slurm-783987"
        assert result.job_id == "783987"
        assert "[FoamNordic] Timing:" in result.longship_log.read_text()
        assert "OpenFOAM=00:00:02" in result.longship_log.read_text()


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_openfoam_library_resolution_preserves_the_exact_binary(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = root / "libfoamnordicOpenFOAM.dylib"
        library.touch()
        _mock_patch_1 = mocker.patch.dict(
            os.environ,
            {"FOAMNORDIC_OPENFOAM_LIB": str(root)},
            clear=False,
        )
        try:
            assert _openfoam_library() == library.resolve()
        finally:
            mocker.stop(_mock_patch_1)


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_openfoam_library_prefers_a_collision_proof_macos_build(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "libfoamnordicOpenFOAM.dylib").touch()
        unique = root / "libfoamnordicOpenFOAM-0123456789ab.dylib"
        unique.touch()
        _mock_patch_2 = mocker.patch.dict(
            os.environ,
            {"FOAMNORDIC_OPENFOAM_LIB": str(root)},
            clear=False,
        )
        try:
            assert _openfoam_library() == unique.resolve()
        finally:
            mocker.stop(_mock_patch_2)


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_openfoam_library_resolves_from_case_toolchain_abi(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = root / "lib/libfoamnordicOpenFOAM.so"
        library.parent.mkdir()
        library.touch()
        case = fno.OpenFOAM.Case(
            case_dir=root / "case",
            run_dir=root / "runs",
            of_cmd="module load openfoam/2512",
        )
        longship = fno.Longship(case=case)
        _mock_patch_3 = mocker.patch.dict(os.environ, {}, clear=True)
        _mock_patch_4 = mocker.patch(
            "foamnordic.execution.launch.toolchain_runtime_candidates",
            return_value=(root,),
        )
        try:
            assert _openfoam_library(longship) == library.resolve()
        finally:
            mocker.stop(_mock_patch_4)
            mocker.stop(_mock_patch_3)


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_submission_environment_does_not_mutate_or_inherit_slurm(mocker) -> None:
    inherited = {
        "SLURM_JOB_ID": "123",
        "SBATCH_MEM_PER_NODE": "4G",
        "SRUN_CPUS_PER_TASK": "1",
        "FOAMNORDIC_KEEP": "yes",
    }
    _mock_patch_5 = mocker.patch.dict(os.environ, inherited, clear=True)
    try:
        environment = _submission_environment()
        assert os.environ["SLURM_JOB_ID"] == "123"
    finally:
        mocker.stop(_mock_patch_5)
    assert environment == {"FOAMNORDIC_KEEP": "yes"}


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o750)


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_launch_prepares_isolated_case_and_returns_run(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        workspace = root / "workspace"
        tools = root / "tools"
        tools.mkdir()
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
            path.write_text("original\n", encoding="utf-8")
        write_mesh(source)
        artifact = root / "nutFjord.fnom"
        fno.export.onnx(
            b"fixture",
            path=artifact,
            inputs={
                "velocity_grad": fno.Tensor.tensor(),
                "filter_width": fno.Tensor.scalar(),
            },
            outputs={"eddy_viscosity": fno.Tensor.scalar()},
        )
        library = root / "lib"
        library.mkdir()
        (library / "libfoamnordicOpenFOAM.so").touch()
        worker = tools / "worker"
        _write_executable(
            worker,
            "#!/bin/sh\n"
            'test -z "${FOAMNORDIC_SOLVER_ONLY:-}" || exit 77\n'
            "ready=''\n"
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = --ready-file ]; then ready=$2; shift 2; else shift; fi\n'
            "done\n"
            'touch "$ready"\n'
            "trap 'rm -f \"$ready\"; exit 0' TERM INT\n"
            "while :; do sleep 1; done\n",
        )
        _write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
        _write_executable(
            tools / "pimpleFoam",
            '#!/bin/sh\ntest "${FOAMNORDIC_SOLVER_ONLY:-}" = 1 || exit 78\n',
        )
        longship = fno.Longship(
            name="launch-test",
            case=fno.openfoam.Case(
                case_dir=source,
                run_dir=workspace,
                of_cmd=(f"export FOAMNORDIC_SOLVER_ONLY=1; export PATH={tools}:$PATH"),
            ),
            closures=(
                fno.Closure(
                    name="nutFjord",
                    artifact=artifact,
                ),
            ),
        )
        environment = {
            "FOAMNORDIC_MODEL_WORKER": str(worker),
            "FOAMNORDIC_OPENFOAM_LIB": str(library),
        }
        _mock_patch_6 = mocker.patch.dict(os.environ, environment, clear=False)
        try:
            run = longship.launch(
                readiness_timeout=2,
                termination_grace=0.1,
                verbose=False,
            )
            result = run.stop(timeout=4)
        finally:
            mocker.stop(_mock_patch_6)
        assert result.success
        assert generated_kind(result.work_dir) == "run"
        assert (source / "constant/turbulenceProperties").read_text() == "original\n"
        rendered = (result.work_dir / "case/constant/turbulenceProperties").read_text()
        assert "LESModel        nutFjord" in rendered
        manifest = json.loads(
            (result.work_dir / ".foamnordic-generated.json").read_text()
        )
        assert manifest["metadata"]["plan_digest"] == result.plan_digest
        assert manifest["metadata"]["plan"]["name"] == "launch-test"
        assert not (result.work_dir / "plan.json").exists()
        assert re.search(
            r"^launch-test-local-\d{8}T\d{6}-[0-9a-f]{6}$", result.work_dir.name
        )
        identity = result.work_dir.name.removeprefix("launch-test-")
        assert result.solver_log.name == f"Sailing_launch-test_{identity}.out"
        assert result.longship_log.name == f"Sailing_launch-test_{identity}.log"
        assert result.host_log.name == f"Harbor_launch-test_{identity}.log"
        assert "███████╗" in result.longship_log.read_text()
        assert "███████╗" in result.host_log.read_text()
        assert "[FoamNordic] Harbor: launch-test" in result.host_log.read_text()
        assert "[FoamNordic] Timing:" in result.longship_log.read_text()
        assert (
            "[FoamNordic] Preparing isolated OpenFOAM case"
            in result.longship_log.read_text()
        )
        assert not any(result.work_dir.rglob("prepare.log"))
        assert result.longship_log.parent.name == "logs"
        assert {
            path.name
            for path in result.work_dir.iterdir()
            if not path.name.startswith(".")
        } == {"case", "logs"}


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_solver_only_launch_preserves_turbulence_dictionary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        tools = root / "tools"
        tools.mkdir()
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
            path.write_text("baseline\n", encoding="utf-8")
        write_mesh(source)
        _write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
        _write_executable(tools / "pimpleFoam", "#!/bin/sh\nexit 0\n")
        run = fno.Longship(
            name="baseline",
            case=fno.openfoam.Case(
                case_dir=source,
                run_dir=root / "workspace",
                of_cmd=f"export PATH={tools}:$PATH",
            ),
        ).launch(verbose=False)
        result = run.stop(timeout=3)
        assert result.success
        assert (
            result.work_dir / "case/constant/turbulenceProperties"
        ).read_text() == "baseline\n"
        assert result.summary(display=False).name == "baseline"
        identity = result.work_dir.name.removeprefix("baseline-")
        assert result.solver_log.name == f"Sailing_baseline_{identity}.out"
        assert "succeeded" in result.longship_log.read_text()
        assert "███████╗" in result.host_log.read_text()


@pytest.mark.skipif(
    not runtime_available(), reason="binary wheel runtime is not installed"
)
def test_zero_orig_is_copied_only_into_the_isolated_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        tools = root / "tools"
        tools.mkdir()
        for relative in (
            "0.orig/U",
            "0.orig/p",
            "constant/turbulenceProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
        ):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("zero-orig\n", encoding="utf-8")
        write_mesh(source)
        _write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
        _write_executable(tools / "pimpleFoam", "#!/bin/sh\nexit 0\n")
        case = fno.OpenFOAM.Case(
            case_dir=source,
            run_dir=root / "workspace",
            of_cmd=f"export PATH={tools}:$PATH",
        )
        assert case.initial_directory == source / "0.orig"
        result = fno.Longship(case=case).launch(verbose=False).stop(timeout=3)
        assert result.success
        assert not (source / "0").exists()
        assert (result.work_dir / "case/0/U").read_text(
            encoding="utf-8"
        ) == "zero-orig\n"
        assert (result.work_dir / "case/0.orig/U").is_file()
