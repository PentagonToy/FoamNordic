from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from foamnordic.execution.observe import ObservationStream
from foamnordic.execution.run import (
    _OpenFOAMProgress,
    RunStatus,
    _launch_local,
    _launch_process,
    _longship_executable,
    _normalize_slurm_state,
    _query_slurm,
    _query_slurm_estimated_start,
    _sailing_paths,
)


class OpenFOAMProgressTests(unittest.TestCase):
    def test_incrementally_reports_latest_physical_time_and_clears(self) -> None:
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
            self.assertIn("Sailing in OpenFOAM: t = 0.001", rendered)
            self.assertIn("Sailing in OpenFOAM: t = 0.002", rendered)
            self.assertFalse(output.read_text(encoding="utf-8").endswith("elapsed"))

    def test_steady_solver_iteration_is_a_fallback(self) -> None:
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "Sailing_simpleFoam.out"
            output.write_text("Iteration = 12\n", encoding="utf-8")
            stream = StringIO()
            monitor = _OpenFOAMProgress(output, started=time.monotonic(), stream=stream)
            monitor.refresh(final=True)
            self.assertIn("Sailing in OpenFOAM: iteration = 12", stream.getvalue())


class ObservationProgressTests(unittest.TestCase):
    def test_collection_reports_exchange_progress_and_clears(self) -> None:
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
            run = Mock(_work_dir=root)
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

            self.assertEqual(len(records), 1)
            self.assertIn(
                "Observing OpenFOAM: t = 0.125 | exchange = 17",
                output.getvalue(),
            )
            self.assertTrue(output.getvalue().endswith("\r"))

    def test_progress_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "boolean"):
            ObservationStream(
                Mock(),
                Path("observations.jsonl"),
                poll_interval=0.1,
                progress="yes",
            )


def packaged_longship_available() -> bool:
    try:
        _longship_executable()
    except RuntimeError:
        return False
    return True


class SlurmMetadataTests(unittest.TestCase):
    def test_longship_resolves_beside_editable_native_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "foamnordic"
            executable = package / "bin/foamnordic-longship"
            executable.parent.mkdir(parents=True)
            executable.touch(mode=0o700)
            native = package / "_native.so"
            native.touch()
            with (
                patch.dict(os.environ, {}, clear=False),
                patch(
                    "foamnordic.execution.run.__file__",
                    str(
                        Path(directory)
                        / "installed/foamnordic/execution/run.py"
                    ),
                ),
                patch(
                    "foamnordic.execution.run.find_spec",
                    return_value=Mock(origin=str(native)),
                ),
            ):
                os.environ.pop("FOAMNORDIC_LONGSHIP", None)
                self.assertEqual(_longship_executable(), executable.resolve())

    def test_accounting_exposes_actual_start_timestamp(self) -> None:
        completed = Mock(
            stdout=(
                "783528|RUNNING|small|00:00:04|rc5130|"
                "2026-08-22T15:42:10\n"
            )
        )
        with (
            patch("foamnordic.execution.run.shutil.which", return_value="/usr/bin/sacct"),
            patch("foamnordic.execution.run.subprocess.run", return_value=completed),
        ):
            details = _query_slurm("783528")
        self.assertEqual(details["status"], "running")
        self.assertEqual(details["start"], "2026-08-22T15:42:10")

    def test_pending_job_exposes_slurm_estimated_start(self) -> None:
        completed = Mock(stdout="2026-08-22T16:10:00\n")
        with (
            patch("foamnordic.execution.run.shutil.which", return_value="/usr/bin/squeue"),
            patch("foamnordic.execution.run.subprocess.run", return_value=completed),
        ):
            self.assertEqual(
                _query_slurm_estimated_start("783528"),
                "2026-08-22T16:10:00",
            )


@unittest.skipUnless(packaged_longship_available(), "Longship executable is not installed")
class RunTests(unittest.TestCase):
    def test_slurm_transitional_states_do_not_look_failed(self) -> None:
        self.assertEqual(_normalize_slurm_state("configuring"), "pending")
        self.assertEqual(_normalize_slurm_state("completing"), "running")
        self.assertEqual(_normalize_slurm_state("completed"), "succeeded")
        self.assertEqual(_normalize_slurm_state("cancelled"), "cancelled")
        self.assertEqual(_normalize_slurm_state("out_of_memory"), "failed")

    def launch(self, directory: str, *, host_script: str, solver_script: str):
        work_dir = Path(directory)
        ready = work_dir / "host.ready"
        host = ("/bin/sh", "-c", host_script, "sh", str(ready))
        solver = ("/bin/sh", "-c", solver_script)
        return _launch_local(
            host=host,
            solver=solver,
            ready_files=(ready,),
            work_dir=work_dir,
            termination_grace=0.1,
            plan_digest="sha256:test",
        ), ready

    def test_stop_returns_successful_durable_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, ready = self.launch(
                directory,
                host_script='touch "$1"; while :; do sleep 1; done',
                solver_script="exit 0",
            )
            result = run.stop(timeout=3)
            self.assertTrue(result.success)
            self.assertIs(result, run.stop())
            self.assertFalse(hasattr(run, "wait"))
            self.assertFalse(hasattr(run, "cancel"))
            self.assertFalse(hasattr(result, "raise_for_status"))
            self.assertEqual(run.status, RunStatus.SUCCEEDED)
            self.assertEqual(result.plan_digest, "sha256:test")
            self.assertFalse(ready.exists())
            self.assertEqual(result.case, result.work_dir / "case")
            self.assertEqual(result.logs, result.work_dir / "logs")
            self.assertEqual(result.artifacts.root, result.work_dir)
            self.assertEqual(result.artifacts.observations, result.work_dir / "observations")
            self.assertEqual(result.artifacts.slurm, result.work_dir / "slurm")

    def test_solver_failure_is_a_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; while :; do sleep 1; done',
                solver_script="exit 7",
            )
            result = run.stop(timeout=3)
            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertEqual(result.exit_code, 7)
            self.assertFalse(result.success)
            self.assertEqual(result.summary(display=False).status, "failed")

    def test_force_stop_terminates_components_and_cleans_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, ready = self.launch(
                directory,
                host_script='touch "$1"; exec sleep 30',
                solver_script="exec sleep 30",
            )
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(0.01)
            result = run.stop(force=True, timeout=3)
            self.assertEqual(result.status, RunStatus.CANCELLED)
            self.assertEqual(result.exit_code, 130)
            self.assertFalse(ready.exists())

    def test_stop_timeout_does_not_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; exec sleep 30',
                solver_script="exec sleep 30",
            )
            with self.assertRaises(TimeoutError):
                run.stop(timeout=0.01)
            self.assertEqual(run.status, RunStatus.RUNNING)
            run.stop(force=True, timeout=3)

    def test_orderly_shutdown_terminates_an_owned_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, ready = self.launch(
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
            self.assertEqual(result.status, RunStatus.CANCELLED)

    def test_detach_preserves_a_run_during_orderly_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; exec sleep 30',
                solver_script="exec sleep 30",
            )
            self.assertIs(run.detach(), run)
            run._shutdown_owned_process()
            self.assertEqual(run.status, RunStatus.RUNNING)
            run.stop(force=True, timeout=3)

    def test_observe_is_iterable_without_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; while :; do sleep 1; done',
                solver_script="sleep 0.1; exit 0",
            )
            record = {
                "exchange_index": 4,
                "time": 0.25,
                "summary": {"nut": {"min": 0.1, "max": 0.4}},
                "timing": {"closure_wait": 0.02, "evaluate": 0.03},
            }
            observations_dir = Path(directory) / "observations"
            observations_dir.mkdir()
            (observations_dir / "observations.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            observations = list(run.observe(poll_interval=0.01))
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].exchange_index, 4)
            self.assertEqual(observations[0].summary["nut"].minimum, 0.1)
            self.assertEqual(observations[0].timing.evaluate, 0.03)
            self.assertTrue(run.stop().success)

    def test_observe_merges_native_rank_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "host.ready"
            run = _launch_local(
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
                    "timing": {"closure_wait": 0.02, "evaluate": 0.003},
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
                    "timing": {"closure_wait": 0.04, "evaluate": 0.005},
                },
            )
            observations_dir = root / "observations"
            observations_dir.mkdir()
            for rank, record in enumerate(records):
                (observations_dir / f"observations.{rank}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            observations = list(run.observe(poll_interval=0.01))
            self.assertEqual(len(observations), 1)
            summary = observations[0].summary["nut"]
            self.assertEqual(summary.minimum, -2.0)
            self.assertEqual(summary.maximum, 4.0)
            self.assertEqual(summary.mean, 1.25)
            self.assertEqual(summary.l2, 5.0)
            self.assertEqual(summary.count, 16)
            self.assertEqual(observations[0].timing.closure_wait, 0.04)
            self.assertTrue(run.stop().success)

    def test_stop_force_requires_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; exec sleep 30',
                solver_script="exec sleep 30",
            )
            with self.assertRaisesRegex(TypeError, "boolean"):
                run.stop(force="yes")
            result = run.stop(timeout=3, force=True)
            self.assertEqual(result.status, RunStatus.CANCELLED)

    def test_result_summary_supports_compact_and_expanded_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.launch(
                directory,
                host_script='touch "$1"; while :; do sleep 1; done',
                solver_script="exit 0",
            )
            result = run.stop(timeout=3)
            self.assertEqual(result.summary("short", display=False).style, "compact")
            self.assertEqual(result.summary("long", display=False).style, "expanded")
            with self.assertRaisesRegex(ValueError, "summary style"):
                result.summary("wide", display=False)

    def test_scheduler_identity_is_added_to_final_log_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sailing, harbor, output = _sailing_paths(root, "NACA4412")
            harbor.touch()
            output.touch()
            job_file = root / ".foamnordic/job.id"
            job_file.parent.mkdir()
            job_file.write_text("123456\n", encoding="utf-8")
            run = _launch_process(
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
            self.assertEqual(result.longship_log.name, "Sailing_NACA4412_123456.log")
            self.assertEqual(result.solver_log.name, "Sailing_NACA4412_123456.out")
            self.assertEqual(result.host_log.name, "Harbor_NACA4412_123456.log")

    def test_wait_for_start_observes_pending_then_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sailing, harbor, output = _sailing_paths(root, "cavity")
            harbor.touch()
            output.touch()
            job_file = root / ".foamnordic/job.id"
            job_file.parent.mkdir()
            job_file.write_text("654321\n", encoding="utf-8")
            run = _launch_process(
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
            with patch(
                "foamnordic.execution.run._query_slurm",
                side_effect=({"status": "pending"}, {"status": "running"}),
            ):
                self.assertEqual(
                    run._wait_for_start(timeout=2),
                    ("654321", "running"),
                )
            run.stop(force=True, timeout=3)


if __name__ == "__main__":
    unittest.main()
