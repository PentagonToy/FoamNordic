from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import foamnordic as fno
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.resources import resource_values
from foamnordic.execution.node_host import _node_index, _rewrite_ready_file
from foamnordic.execution.slurm import write_batch, write_submission_wrapper


@unittest.skipUnless(native_available(), "nanobind extension is not installed")
class SlurmRenderingTests(unittest.TestCase):
    def test_node_host_uses_slurm_process_index(self) -> None:
        with patch.dict(os.environ, {"SLURM_PROCID": "1"}, clear=False):
            self.assertEqual(_node_index(2), 1)

    def test_node_host_rewrites_each_ready_marker(self) -> None:
        command = [
            "worker",
            "--ready-file",
            "/tmp/first.ready",
            "--ready-file",
            "/tmp/second.ready",
        ]

        self.assertEqual(
            _rewrite_ready_file(command, ".node1"),
            [
                "worker",
                "--ready-file",
                "/tmp/first.ready.node1",
                "--ready-file",
                "/tmp/second.ready.node1",
            ],
        )

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
                openfoam=fno.Slurm.openfoam(
                    nodes=1,
                    ntasks=2,
                    cpus_per_task=1,
                    mem_per_cpu="1G",
                ),
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
        self.assertIn(": >>", script)
        self.assertIn("--ntasks=2", script)
        self.assertIn("#SBATCH --ntasks=2", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertIn("#SBATCH --mem-per-cpu=1G", script)
        self.assertIn("--cpus-per-task=1 \\", script)
        self.assertIn("--cpu-bind=none \\", script)
        self.assertIn("/logs/Sailing_baseline.out", script)
        self.assertIn("/logs/Sailing_baseline.log", script)
        self.assertIn(
            "unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE",
            script,
        )
        self.assertIn("#SBATCH --open-mode=append", script)
        self.assertNotIn("███████╗", script)
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
                (root / ".foamnordic/closure.ready",),
                120.0,
                30.0,
            )
            script = batch.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --ntasks=3", script)
        self.assertIn("--host srun --nodes=1 --ntasks=1", script)
        self.assertIn("--solver srun --nodes=1 --ntasks=2", script)

    def test_explicit_resources_keep_solver_and_model_cpu_shapes_separate(self) -> None:
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

        self.assertEqual(values["solver_cpus"], 16)
        self.assertEqual(values["model_cpus"], 8)
        self.assertEqual(values["shm"], 512 * 1024**2)
        self.assertIn("#SBATCH --ntasks=24", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertEqual(values["total_memory"], int(40.5 * 1024**3))
        self.assertIn("#SBATCH --mem=41472M", script)
        self.assertIn("--host srun --nodes=1 --ntasks=1", script)
        self.assertIn("--cpus-per-task=8 --cpu-bind=none", script)
        self.assertIn("--solver srun --nodes=1 --ntasks=16", script)

    def test_multi_node_batch_waits_for_every_attached_host(self) -> None:
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

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks=20", script)
        self.assertIn("--ready", script)
        self.assertIn("closure.ready.node0", script)
        self.assertIn("closure.ready.node1", script)
        self.assertIn("--host srun --nodes=2 --ntasks=2", script)
        self.assertIn("--solver srun --nodes=2 --ntasks=16", script)
        self.assertIn("--ntasks-per-node=8", script)

    def test_submission_wrapper_owns_scancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = write_submission_wrapper(root, root / "slurm/longship.sbatch")
            script = wrapper.read_text(encoding="utf-8")
        self.assertEqual(wrapper.parent.name, "slurm")
        self.assertIn('scancel "$job_id"', script)
        self.assertIn('read -r -u "$FOAMNORDIC_OWNER_FD"', script)
        self.assertIn('FOAMNORDIC_ORPHAN_TIMEOUT:-30', script)
        self.assertIn(".foamnordic/job.id", script)
        self.assertIn('SLURM_*|SBATCH_*|SRUN_*) unset "$key"', script)
        self.assertIn('"${job_id}.batch" "$job_id"', script)
        self.assertIn("COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*", script)
        self.assertNotIn(
            'while squeue --noheader --job "$job_id"',
            script,
        )
        self.assertNotIn("seq 1 300", script)


class SubmissionWrapperTests(unittest.TestCase):
    def test_owner_pipe_eof_cancels_without_waiting_for_allocation(self) -> None:
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
                    "if [[ -f \"$FNO_CANCELLED\" ]]; then\n"
                    "  printf 'CANCELLED|\\n'\n"
                    "fi\n"
                ),
                "squeue": "#!/bin/bash\nprintf '12345 running\\n'\n",
                "scancel": "#!/bin/bash\n: > \"$FNO_CANCELLED\"\n",
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

        self.assertEqual(process.returncode, 130, stderr)
        self.assertTrue(was_cancelled)

    def test_terminal_batch_step_does_not_wait_for_allocation_epilog(self) -> None:
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
                    "case \"$*\" in\n"
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

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
