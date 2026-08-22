from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import foamnordic as fno
from foamnordic.execution.launch import _openfoam_library, _submission_environment
from foamnordic.core.managed import generated_kind, mark_generated
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.run import _launch_process, _longship_executable
from foamnordic.execution.host_group import main as run_host_group


def runtime_available() -> bool:
    if not native_available():
        return False
    try:
        _longship_executable()
    except RuntimeError:
        return False
    return True


class HostGroupTests(unittest.TestCase):
    def test_multiple_workers_publish_one_group_readiness_marker(self) -> None:
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
            self.assertEqual(run_host_group(str(configuration)), 0)
            self.assertFalse(aggregate.exists())
            self.assertTrue(all(not Path(path).exists() for path in ready_files))


class RunArtifactTests(unittest.TestCase):
    def test_slurm_identity_and_timing_are_finalized_after_completion(self) -> None:
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

            self.assertEqual(result.work_dir.name, "case-name-slurm-783987")
            self.assertEqual(result.job_id, "783987")
            self.assertIn("[FoamNordic] Timing:", result.longship_log.read_text())
            self.assertIn("OpenFOAM=00:00:02", result.longship_log.read_text())


@unittest.skipUnless(runtime_available(), "binary wheel runtime is not installed")
class LaunchTests(unittest.TestCase):
    def test_openfoam_library_resolution_preserves_the_exact_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libfoamnordicOpenFOAM.dylib"
            library.touch()
            with patch.dict(
                os.environ,
                {"FOAMNORDIC_OPENFOAM_LIB": str(root)},
                clear=False,
            ):
                self.assertEqual(_openfoam_library(), library.resolve())

    def test_openfoam_library_resolves_from_case_toolchain_abi(self) -> None:
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
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "foamnordic.execution.launch.toolchain_runtime_candidates",
                    return_value=(root,),
                ),
            ):
                self.assertEqual(_openfoam_library(longship), library.resolve())

    def test_submission_environment_does_not_mutate_or_inherit_slurm(self) -> None:
        inherited = {
            "SLURM_JOB_ID": "123",
            "SBATCH_MEM_PER_NODE": "4G",
            "SRUN_CPUS_PER_TASK": "1",
            "FOAMNORDIC_KEEP": "yes",
        }
        with patch.dict(os.environ, inherited, clear=True):
            environment = _submission_environment()
            self.assertEqual(os.environ["SLURM_JOB_ID"], "123")
        self.assertEqual(environment, {"FOAMNORDIC_KEEP": "yes"})

    def _write_executable(self, path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o750)

    def test_launch_prepares_isolated_case_and_returns_run(self) -> None:
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
            artifact = root / "nutFjord.fnom"
            fno.export.onnx(
                b"fixture",
                path=artifact,
                inputs={"velocity_grad": fno.Tensor.tensor()},
                outputs={"eddy_viscosity": fno.Tensor.scalar()},
            )
            library = root / "lib"
            library.mkdir()
            (library / "libfoamnordicOpenFOAM.so").touch()
            worker = tools / "worker"
            self._write_executable(
                worker,
                "#!/bin/sh\n"
                "test -z \"${FOAMNORDIC_SOLVER_ONLY:-}\" || exit 77\n"
                "ready=''\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = --ready-file ]; then ready=$2; shift 2; else shift; fi\n"
                "done\n"
                "touch \"$ready\"\n"
                "trap 'rm -f \"$ready\"; exit 0' TERM INT\n"
                "while :; do sleep 1; done\n",
            )
            self._write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
            self._write_executable(
                tools / "pimpleFoam",
                "#!/bin/sh\n"
                "test \"${FOAMNORDIC_SOLVER_ONLY:-}\" = 1 || exit 78\n",
            )
            longship = fno.Longship(
                name="launch-test",
                case=fno.openfoam.Case(
                    case_dir=source,
                    run_dir=workspace,
                    of_cmd=(
                        "export FOAMNORDIC_SOLVER_ONLY=1; "
                        f"export PATH={tools}:$PATH"
                    ),
                ),
                closures=(
                    fno.Closure(
                        name="nutFjord",
                        artifact=artifact,
                        inputs={"velocity_grad": fno.grad("U")},
                        outputs={"eddy_viscosity": fno.field("nut")},
                    ),
                ),
            )
            environment = {
                "FOAMNORDIC_CLOSURE_WORKER": str(worker),
                "FOAMNORDIC_OPENFOAM_LIB": str(library),
            }
            with patch.dict(os.environ, environment, clear=False):
                run = longship.launch(
                    readiness_timeout=2,
                    termination_grace=0.1,
                    verbose=False,
                )
                result = run.stop(timeout=4)
            self.assertTrue(result.success)
            self.assertEqual(generated_kind(result.work_dir), "run")
            self.assertEqual(
                (source / "constant/turbulenceProperties").read_text(), "original\n"
            )
            rendered = (
                result.work_dir / "case/constant/turbulenceProperties"
            ).read_text()
            self.assertIn("LESModel        nutFjord", rendered)
            manifest = json.loads(
                (result.work_dir / ".foamnordic-generated.json").read_text()
            )
            self.assertEqual(manifest["metadata"]["plan_digest"], result.plan_digest)
            self.assertEqual(manifest["metadata"]["plan"]["name"], "launch-test")
            self.assertFalse((result.work_dir / "plan.json").exists())
            self.assertRegex(
                result.work_dir.name,
                r"^launch-test-local-\d{8}T\d{6}-[0-9a-f]{6}$",
            )
            identity = result.work_dir.name.removeprefix("launch-test-")
            self.assertEqual(
                result.solver_log.name, f"Sailing_launch-test_{identity}.out"
            )
            self.assertEqual(
                result.longship_log.name, f"Sailing_launch-test_{identity}.log"
            )
            self.assertEqual(
                result.host_log.name, f"Harbor_launch-test_{identity}.log"
            )
            self.assertIn("███████╗", result.longship_log.read_text())
            self.assertIn("███████╗", result.host_log.read_text())
            self.assertIn(
                "[FoamNordic] Harbor: launch-test",
                result.host_log.read_text(),
            )
            self.assertIn("[FoamNordic] Timing:", result.longship_log.read_text())
            self.assertEqual(result.longship_log.parent.name, "logs")
            self.assertEqual(
                {path.name for path in result.work_dir.iterdir() if not path.name.startswith(".")},
                {"case", "logs"},
            )

    def test_solver_only_launch_preserves_turbulence_dictionary(self) -> None:
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
            self._write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
            self._write_executable(tools / "pimpleFoam", "#!/bin/sh\nexit 0\n")
            run = fno.Longship(
                name="baseline",
                case=fno.openfoam.Case(
                    case_dir=source,
                    run_dir=root / "workspace",
                    of_cmd=f"export PATH={tools}:$PATH",
                ),
            ).launch(verbose=False)
            result = run.stop(timeout=3)
            self.assertTrue(result.success)
            self.assertEqual(
                (result.work_dir / "case/constant/turbulenceProperties").read_text(),
                "baseline\n",
            )
            self.assertEqual(result.summary(display=False).name, "baseline")
            identity = result.work_dir.name.removeprefix("baseline-")
            self.assertEqual(result.solver_log.name, f"Sailing_baseline_{identity}.out")
            self.assertIn("succeeded", result.longship_log.read_text())
            self.assertIn("███████╗", result.host_log.read_text())

    def test_zero_orig_is_copied_only_into_the_isolated_run(self) -> None:
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
            self._write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
            self._write_executable(tools / "pimpleFoam", "#!/bin/sh\nexit 0\n")
            case = fno.OpenFOAM.Case(
                case_dir=source,
                run_dir=root / "workspace",
                of_cmd=f"export PATH={tools}:$PATH",
            )
            self.assertEqual(case.initial_directory, source / "0.orig")
            result = fno.Longship(case=case).launch(verbose=False).stop(timeout=3)
            self.assertTrue(result.success)
            self.assertFalse((source / "0").exists())
            self.assertEqual(
                (result.work_dir / "case/0/U").read_text(encoding="utf-8"),
                "zero-orig\n",
            )
            self.assertTrue((result.work_dir / "case/0.orig/U").is_file())


if __name__ == "__main__":
    unittest.main()
