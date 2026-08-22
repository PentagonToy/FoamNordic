from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import foamnordic as fno
from foamnordic._launch import _submission_environment
from foamnordic._managed import generated_kind
from foamnordic._native_plan import available as native_available
from foamnordic._run import _longship_executable


def runtime_available() -> bool:
    if not native_available():
        return False
    try:
        _longship_executable()
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(runtime_available(), "binary wheel runtime is not installed")
class LaunchTests(unittest.TestCase):
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
                "ready=''\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = --ready-file ]; then ready=$2; shift 2; else shift; fi\n"
                "done\n"
                "touch \"$ready\"\n"
                "trap 'rm -f \"$ready\"; exit 0' TERM INT\n"
                "while :; do sleep 1; done\n",
            )
            self._write_executable(tools / "foamDictionary", "#!/bin/sh\nexit 0\n")
            self._write_executable(tools / "pimpleFoam", "#!/bin/sh\nexit 0\n")
            longship = fno.Longship(
                name="launch-test",
                case=fno.openfoam.Case(
                    case_dir=source,
                    run_dir=workspace,
                    of_cmd=f"export PATH={tools}:$PATH",
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
            identity = f"local-{run.pid}"
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
            self.assertEqual(
                result.solver_log.name, f"Sailing_baseline_local-{run.pid}.out"
            )
            self.assertIn("succeeded", result.longship_log.read_text())

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
