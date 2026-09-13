from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
from foamnordic.execution.observe import ObservationStream
from foamnordic.execution.process import (
    container_private_path,
    launch_local,
    launch_process,
    longship_executable,
)
from foamnordic.execution.run import (
    RunStatus,
    _normalize_slurm_state,
    _OpenFOAMProgress,
    _query_slurm,
    _query_slurm_estimated_start,
)
from foamnordic.execution.run_files import sailing_paths


def test_incrementally_reports_latest_physical_time_and_clears() -> None:
    from io import StringIO

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "Sailing_cavity.out"
        stream = StringIO()
        monitor = _OpenFOAMProgress(output, started=time.monotonic(), stream=stream)

        output.write_text("Create time\nTime = 0.001\n", encoding="utf-8")
        monitor.refresh()
        with output.open("a", encoding="utf-8") as log:
            log.write("PIMPLE: iteration 1\nTime = 0.002\n")
        monitor.refresh()
        monitor.clear()

        rendered = stream.getvalue()
        assert "Sailing in OpenFOAM: t = 0.001" in rendered
        assert "Sailing in OpenFOAM: t = 0.002" in rendered
        assert not output.read_text(encoding="utf-8").endswith("elapsed")


def test_steady_solver_iteration_is_a_fallback() -> None:
    from io import StringIO

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "Sailing_simpleFoam.out"
        output.write_text("Iteration = 12\n", encoding="utf-8")
        stream = StringIO()
        monitor = _OpenFOAMProgress(output, started=time.monotonic(), stream=stream)
        monitor.refresh(final=True)
        assert "Sailing in OpenFOAM: iteration = 12" in stream.getvalue()


def test_collection_reports_exchange_progress_and_clears(mocker) -> None:
    from io import StringIO

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        observations = root / "observations"
        observations.mkdir()
        path = observations / "observations.jsonl"
        path.write_text(
            json.dumps(
                {
                    "exchange_index": 17,
                    "time": 0.125,
                    "summary": {"U": {"l2": 2.5}},
                    "timing": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = mocker.Mock(_work_dir=root)
        run.status.value = "succeeded"
        output = StringIO()
        stream = ObservationStream(
            run,
            path,
            poll_interval=0.001,
            progress=True,
            progress_stream=output,
        )

        records = list(stream)

        assert len(records) == 1
        assert "Observing OpenFOAM: t = 0.125 | exchange = 17" in output.getvalue()
        assert output.getvalue().endswith("\r")


def test_progress_requires_a_boolean(mocker) -> None:
    with pytest.raises(TypeError, match="boolean"):
        ObservationStream(
            mocker.Mock(),
            Path("observations.jsonl"),
            poll_interval=0.1,
            progress="yes",
        )


def packaged_longship_available() -> bool:
    try:
        longship_executable()
    except RuntimeError:
        return False
    return True


def test_tykky_build_root_is_container_private() -> None:
    assert container_private_path(
        Path("/ROIHU_TYKKY_example/miniforge/envs/env1/bin/foamnordic-longship")
    )


def test_longship_prefers_abi_runtime_over_packaged_executable(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = root / "runtime"
        executable = runtime / "bin/foamnordic-longship"
        executable.parent.mkdir(parents=True)
        executable.touch(mode=0o700)
        package = root / "package/foamnordic/_native.so"
        package.parent.mkdir(parents=True)
        package.touch()
        _mock_patch_1 = mocker.patch.dict(os.environ, {}, clear=False)
        _mock_patch_2 = mocker.patch(
            "foamnordic.execution.process.active_runtime_candidates",
            return_value=(runtime,),
        )
        _mock_patch_3 = mocker.patch(
            "foamnordic.execution.process.find_spec",
            return_value=mocker.Mock(origin=str(package)),
        )
        try:
            os.environ.pop("FOAMNORDIC_LONGSHIP", None)
            assert longship_executable() == executable.resolve()
        finally:
            mocker.stop(_mock_patch_3)
            mocker.stop(_mock_patch_2)
            mocker.stop(_mock_patch_1)


def test_longship_resolves_beside_editable_native_module(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "foamnordic"
        executable = package / "bin/foamnordic-longship"
        executable.parent.mkdir(parents=True)
        executable.touch(mode=0o700)
        native = package / "_native.so"
        native.touch()
        _mock_patch_4 = mocker.patch.dict(os.environ, {}, clear=False)
        _mock_patch_5 = mocker.patch(
            "foamnordic.execution.process.__file__",
            str(Path(directory) / "installed/foamnordic/execution/process.py"),
        )
        _mock_patch_6 = mocker.patch(
            "foamnordic.execution.process.find_spec",
            return_value=mocker.Mock(origin=str(native)),
        )
        _mock_patch_7 = mocker.patch(
            "foamnordic.execution.process.active_runtime_candidates",
            return_value=(),
        )
        try:
            os.environ.pop("FOAMNORDIC_LONGSHIP", None)
            assert longship_executable() == executable.resolve()
        finally:
            mocker.stop(_mock_patch_7)
            mocker.stop(_mock_patch_6)
            mocker.stop(_mock_patch_5)
            mocker.stop(_mock_patch_4)


def test_accounting_exposes_actual_start_timestamp(mocker) -> None:
    completed = mocker.Mock(
        stdout=("783528|RUNNING|small|00:00:04|rc5130|2026-08-22T15:42:10\n")
    )
    _mock_patch_8 = mocker.patch(
        "foamnordic.execution.run.shutil.which", return_value="/usr/bin/sacct"
    )
    _mock_patch_9 = mocker.patch(
        "foamnordic.execution.run.subprocess.run", return_value=completed
    )
    try:
        details = _query_slurm("783528")
    finally:
        mocker.stop(_mock_patch_9)
        mocker.stop(_mock_patch_8)
    assert details["status"] == "running"
    assert details["start"] == "2026-08-22T15:42:10"


def test_pending_job_exposes_slurm_estimated_start(mocker) -> None:
    completed = mocker.Mock(stdout="2026-08-22T16:10:00\n")
    _mock_patch_10 = mocker.patch(
        "foamnordic.execution.run.shutil.which", return_value="/usr/bin/squeue"
    )
    _mock_patch_11 = mocker.patch(
        "foamnordic.execution.run.subprocess.run", return_value=completed
    )
    try:
        assert _query_slurm_estimated_start("783528") == "2026-08-22T16:10:00"
    finally:
        mocker.stop(_mock_patch_11)
        mocker.stop(_mock_patch_10)


def test_pending_job_falls_back_to_scontrol_start_time(mocker) -> None:
    def executable(name):
        return f"/usr/bin/{name}"

    results = [
        mocker.Mock(stdout="N/A\n"),
        mocker.Mock(
            stdout=(
                "JobId=783528 JobState=PENDING "
                "StartTime=2026-08-22T16:10:00 Reason=Resources\n"
            )
        ),
    ]
    _mock_patch_12 = mocker.patch(
        "foamnordic.execution.run.shutil.which", side_effect=executable
    )
    _mock_patch_13 = mocker.patch(
        "foamnordic.execution.run.subprocess.run", side_effect=results
    )
    try:
        assert _query_slurm_estimated_start("783528") == "2026-08-22T16:10:00"
    finally:
        mocker.stop(_mock_patch_13)
        mocker.stop(_mock_patch_12)


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_slurm_transitional_states_do_not_look_failed() -> None:
    assert _normalize_slurm_state("configuring") == "pending"
    assert _normalize_slurm_state("completing") == "running"
    assert _normalize_slurm_state("completed") == "succeeded"
    assert _normalize_slurm_state("cancelled") == "cancelled"
    assert _normalize_slurm_state("out_of_memory") == "failed"


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def launch(directory: str, *, host_script: str, solver_script: str):
    work_dir = Path(directory)
    ready = work_dir / "host.ready"
    host = ("/bin/sh", "-c", host_script, "sh", str(ready))
    solver = ("/bin/sh", "-c", solver_script)
    return launch_local(
        host=host,
        solver=solver,
        ready_files=(ready,),
        work_dir=work_dir,
        termination_grace=0.1,
        plan_digest="sha256:test",
    ), ready


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_detach_releases_the_submission_owner_pipe() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "owner.log"
        run = launch_process(
            (
                sys.executable,
                "-c",
                (
                    "import os; "
                    "fd = int(os.environ['FOAMNORDIC_OWNER_FD']); "
                    "print(os.read(fd, 64).decode(), end='')"
                ),
            ),
            work_dir=root,
            process_log=output,
            longship_log=root / "longship.log",
            host_log=root / "host.log",
            solver_log=root / "solver.log",
            environment=os.environ,
            orphan_timeout=0,
        )

        run.detach()
        result = run.stop(timeout=3)

        assert result.success
        assert output.read_text(encoding="utf-8") == "detach\n"


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_stop_returns_successful_durable_result() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, ready = launch(
            directory,
            host_script='touch "$1"; while :; do sleep 1; done',
            solver_script="exit 0",
        )
        result = run.stop(timeout=3)
        assert result.success
        assert result is run.stop()
        assert not hasattr(run, "wait")
        assert not hasattr(run, "cancel")
        assert not hasattr(result, "raise_for_status")
        assert run.status == RunStatus.SUCCEEDED
        assert result.plan_digest == "sha256:test"
        assert not ready.exists()
        assert result.case == result.work_dir / "case"
        assert result.logs == result.work_dir / "logs"
        assert result.artifacts.root == result.work_dir
        assert result.artifacts.observations == result.work_dir / "observations"
        assert result.artifacts.slurm == result.work_dir / "slurm"


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_solver_failure_is_a_failed_result() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; while :; do sleep 1; done',
            solver_script="exit 7",
        )
        result = run.stop(timeout=3)
        assert result.status == RunStatus.FAILED
        assert result.exit_code == 7
        assert not result.success
        assert result.summary(display=False).status == "failed"


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_force_stop_terminates_components_and_cleans_readiness() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, ready = launch(
            directory,
            host_script='touch "$1"; exec sleep 30',
            solver_script="exec sleep 30",
        )
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        result = run.stop(force=True, timeout=3)
        assert result.status == RunStatus.CANCELLED
        assert result.exit_code == 130
        assert not ready.exists()


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_stop_timeout_does_not_cancel() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; exec sleep 30',
            solver_script="exec sleep 30",
        )
        with pytest.raises(TimeoutError):
            run.stop(timeout=0.01)
        assert run.status == RunStatus.RUNNING
        run.stop(force=True, timeout=3)


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_orderly_shutdown_terminates_an_owned_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, ready = launch(
            directory,
            host_script='touch "$1"; exec sleep 30',
            solver_script="exec sleep 30",
        )
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        run._shutdown_owned_process()
        result = run.stop(timeout=3)
        assert result.status == RunStatus.CANCELLED


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_detach_preserves_a_run_during_orderly_shutdown() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; exec sleep 30',
            solver_script="exec sleep 30",
        )
        assert run.detach() is run
        run._shutdown_owned_process()
        assert run.status == RunStatus.RUNNING
        run.stop(force=True, timeout=3)


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_observe_is_iterable_without_context_manager() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; while :; do sleep 1; done',
            solver_script="sleep 0.1; exit 0",
        )
        record = {
            "exchange_index": 4,
            "time": 0.25,
            "summary": {"nut": {"min": 0.1, "max": 0.4}},
            "timing": {"model_wait": 0.02, "evaluate": 0.03},
        }
        observations_dir = Path(directory) / "observations"
        observations_dir.mkdir()
        (observations_dir / "observations.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        observations = list(run.observe(poll_interval=0.01))
        assert len(observations) == 1
        assert observations[0].exchange_index == 4
        assert observations[0].summary["nut"].minimum == 0.1
        assert observations[0].timing.evaluate == 0.03
        assert run.stop().success


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_observe_merges_native_rank_shards() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ready = root / "host.ready"
        run = launch_local(
            host=(
                "/bin/sh",
                "-c",
                'touch "$1"; while :; do sleep 1; done',
                "sh",
                str(ready),
            ),
            solver=("/bin/sh", "-c", "sleep 0.1; exit 0"),
            ready_files=(ready,),
            work_dir=root,
            termination_grace=0.1,
            observation_sources=2,
        )
        records = (
            {
                "exchange_index": 8,
                "time": 0.5,
                "summary": {
                    "nut": {
                        "minimum": -1.0,
                        "maximum": 2.0,
                        "mean": 0.5,
                        "l2": 3.0,
                        "count": 4,
                    }
                },
                "timing": {"model_wait": 0.02, "evaluate": 0.003},
            },
            {
                "exchange_index": 8,
                "time": 0.5,
                "summary": {
                    "nut": {
                        "minimum": -2.0,
                        "maximum": 4.0,
                        "mean": 1.5,
                        "l2": 4.0,
                        "count": 12,
                    }
                },
                "timing": {"model_wait": 0.04, "evaluate": 0.005},
            },
        )
        observations_dir = root / "observations"
        observations_dir.mkdir()
        for rank, record in enumerate(records):
            (observations_dir / f"observations.{rank}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        observations = list(run.observe(poll_interval=0.01))
        assert len(observations) == 1
        summary = observations[0].summary["nut"]
        assert summary.minimum == -2.0
        assert summary.maximum == 4.0
        assert summary.mean == 1.25
        assert summary.l2 == 5.0
        assert summary.count == 16
        assert observations[0].timing.model_wait == 0.04
        assert run.stop().success


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_stop_force_requires_boolean() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; exec sleep 30',
            solver_script="exec sleep 30",
        )
        with pytest.raises(TypeError, match="boolean"):
            run.stop(force="yes")
        result = run.stop(timeout=3, force=True)
        assert result.status == RunStatus.CANCELLED


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_result_summary_supports_compact_and_expanded_aliases() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run, _ = launch(
            directory,
            host_script='touch "$1"; while :; do sleep 1; done',
            solver_script="exit 0",
        )
        result = run.stop(timeout=3)
        assert result.summary("short", display=False).style == "compact"
        assert result.summary("long", display=False).style == "expanded"
        with pytest.raises(ValueError, match="summary style"):
            result.summary("wide", display=False)


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_scheduler_identity_is_added_to_final_log_names() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sailing, harbor, output = sailing_paths(root, "NACA4412")
        harbor.touch()
        output.touch()
        job_file = root / ".foamnordic/job.id"
        job_file.parent.mkdir()
        job_file.write_text("123456\n", encoding="utf-8")
        run = launch_process(
            ("/bin/sh", "-c", "exit 0"),
            work_dir=root,
            process_log=root / ".foamnordic/submission.log",
            longship_log=sailing,
            host_log=harbor,
            solver_log=output,
            name="NACA4412",
            job_file=job_file,
            partition="small",
        )
        result = run.stop(timeout=3)
        assert result.longship_log.name == "Sailing_NACA4412_123456.log"
        assert result.solver_log.name == "Sailing_NACA4412_123456.out"
        assert result.host_log.name == "Harbor_NACA4412_123456.log"


@pytest.mark.skipif(
    not packaged_longship_available(), reason="Longship executable is not installed"
)
def test_wait_for_start_observes_pending_then_running(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sailing, harbor, output = sailing_paths(root, "cavity")
        harbor.touch()
        output.touch()
        job_file = root / ".foamnordic/job.id"
        job_file.parent.mkdir()
        job_file.write_text("654321\n", encoding="utf-8")
        run = launch_process(
            ("/bin/sh", "-c", "exec sleep 30"),
            work_dir=root,
            process_log=root / ".foamnordic/submission.log",
            longship_log=sailing,
            host_log=harbor,
            solver_log=output,
            name="cavity",
            job_file=job_file,
            partition="small",
        )
        _mock_patch_14 = mocker.patch(
            "foamnordic.execution.run._query_slurm",
            side_effect=({"status": "pending"}, {"status": "running"}),
        )
        try:
            assert run._wait_for_start(timeout=2) == ("654321", "running")
        finally:
            mocker.stop(_mock_patch_14)
        run.stop(force=True, timeout=3)
