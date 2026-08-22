from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import foamnordic as fno
from foamnordic._native_plan import available as native_available
from foamnordic._slurm import write_batch, write_submission_wrapper


@unittest.skipUnless(native_available(), "nanobind extension is not installed")
class SlurmRenderingTests(unittest.TestCase):
    def _longship(self, *, coupled: bool = False) -> fno.Longship:
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
                nodes=1,
                ntasks=2,
                cpus_per_task=1,
                mem_per_cpu="1G",
            ),
        )

    def test_solver_only_batch_has_no_closure_host(self) -> None:
        longship = self._longship()
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
        self.assertNotIn("foamnordic-longship", script)
        self.assertNotIn("--host srun", script)
        self.assertIn("--ntasks=2", script)
        self.assertIn("#SBATCH --ntasks=2", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertIn("#SBATCH --mem-per-cpu=1G", script)
        self.assertIn("--cpus-per-task=1 \\", script)
        self.assertIn("--cpu-bind=none \\", script)
        self.assertIn("/logs/Sailing_baseline.out", script)
        self.assertIn("/logs/Sailing_baseline.log", script)
        self.assertIn(
            "unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU",
            script,
        )
        self.assertIn("███████╗", script)
        self.assertEqual(batch.parent.name, "slurm")

    def test_coupled_batch_reserves_native_sidecar_task(self) -> None:
        longship = self._longship(coupled=True)
        runtime = longship.compile().as_dict()["runtime"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = write_batch(
                longship,
                runtime,
                root,
                ("closure-host",),
                ("pimpleFoam", "-parallel"),
                root / ".foamnordic/closure.ready",
                120.0,
                30.0,
            )
            script = batch.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --ntasks=3", script)
        self.assertIn("--host srun --nodes=1 --ntasks=1", script)
        self.assertIn("--solver srun --nodes=1 --ntasks=2", script)

    def test_submission_wrapper_owns_scancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = write_submission_wrapper(root, root / "slurm/longship.sbatch")
            script = wrapper.read_text(encoding="utf-8")
        self.assertEqual(wrapper.parent.name, "slurm")
        self.assertIn('scancel "$job_id"', script)
        self.assertIn(".foamnordic/job.id", script)
        self.assertIn('SLURM_*|SBATCH_*|SRUN_*) unset "$key"', script)


if __name__ == "__main__":
    unittest.main()
