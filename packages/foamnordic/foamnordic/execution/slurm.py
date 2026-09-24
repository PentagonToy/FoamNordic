"""Slurm rendering, submission ownership, and force cancellation."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .process import longship_executable
from .resources import SHM_MIB_PER_RANK_PROGRAM, memory_bytes, slurm_memory
from .run_files import internal_path, sailing_paths
from .shell import quote_command
from .templates import read as read_template
from .templates import render as render_template

if TYPE_CHECKING:
    from ..core.spec import Longship


def _template(name: str) -> str:
    return read_template(f"slurm/{name}", kind="Slurm")


def _render(name: str, variables: Mapping[str, object]) -> str:
    return render_template(_template(name), variables, kind="Slurm")


def _solver_mpi_directive(ntasks: int) -> str:
    if ntasks <= 1 or shutil.which("srun") is None:
        return ""
    try:
        completed = subprocess.run(
            ["srun", "--mpi=list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    mpi_types = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and " " not in line.strip()
    }
    return "  --mpi=pmix \\\n" if "pmix" in mpi_types else ""


def write_batch(
    longship: Longship,
    runtime: Mapping[str, object],
    work_dir: Path,
    host: Sequence[str] | None,
    solver: Sequence[str],
    ready: Sequence[Path] | None,
    readiness_timeout: float,
    termination_grace: float,
) -> Path:
    scheduler = longship.scheduler
    assert scheduler is not None
    job_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in longship.name
    )[:128]
    model = scheduler.model_resources
    model_gres = (
        f"gpu:{model.gres_model}:{model.gres_count}"
        if host is not None and model.gres_model is not None
        else None
    )
    gres_directive = (
        "" if model_gres is None else f"#SBATCH --gres={model_gres}\n"
    )
    host_gres = "" if model_gres is None else f"    --gres={model_gres} \\\n"
    solver_gres = "" if model_gres is None else "    --gres=none \\\n"
    if not (
        host is not None
        and scheduler.has_model_resources
        and model.mem_per_cpu is not None
    ):
        memory = (
            ""
            if scheduler.mem_per_cpu is None
            else (
                f"#SBATCH --mem-per-cpu={scheduler.mem_per_cpu}"
                "      # Memory reserved per CPU core\n"
            )
        )
    else:
        tasks_per_node = scheduler.ntasks // scheduler.nodes
        memory_per_cpu = memory_bytes(scheduler.mem_per_cpu)
        model_memory_per_cpu = memory_bytes(model.mem_per_cpu)
        shm_per_node = (
            SHM_MIB_PER_RANK_PROGRAM
            * 1024**2
            * tasks_per_node
            * len(longship.field_programs)
            if host is not None and longship.placement.data_path in {"auto", "shm"}
            else 0
        )
        memory_per_node = (
            memory_per_cpu * tasks_per_node * scheduler.cpus_per_task
            + model_memory_per_cpu * model.cpus_per_task
            + shm_per_node
        )
        memory = (
            ""
            if memory_per_node is None
            else (
                f"#SBATCH --mem={slurm_memory(memory_per_node)}"
                "              # OpenFOAM and model memory per node\n"
            )
        )
    memory_sanitizer = (
        "unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE"
    )
    account_directive = (
        ""
        if scheduler.account is None
        else f"#SBATCH --account={scheduler.account}                # Billing project\n"
    )
    partition_directive = (
        ""
        if scheduler.partition is None
        else (
            f"#SBATCH --partition={scheduler.partition}"
            "            # Job partition (queue)\n"
        )
    )
    longship_log, host_log, solver_log = sailing_paths(work_dir, longship.name)
    solver_mpi = _solver_mpi_directive(scheduler.ntasks)
    if host is None:
        allocation_tasks = scheduler.ntasks
        allocation_cpus_per_task = scheduler.cpus_per_task
        launch_body = f""": >> {quote_command((host_log,))}
exec srun \\
{solver_mpi}  --nodes={scheduler.nodes} \\
  --ntasks={scheduler.ntasks} \\
  --ntasks-per-node={runtime['solver_tasks_per_node']} \\
  --cpus-per-task={scheduler.cpus_per_task} \\
  --cpu-bind=none \\
  --output={quote_command((solver_log,))} \\
  --exact \\
  --exclusive \\
  {quote_command(solver)}"""
    else:
        allocation_tasks = scheduler.nodes
        allocation_cpus_per_task = int(runtime["allocation_cpus_per_node"])
        if not ready:
            raise ValueError("coupled Slurm launch requires readiness markers")
        ready_arguments = " \\\n".join(
            f"  --ready {quote_command((path,))}" for path in ready
        )
        launch_body = f"""exec {quote_command((longship_executable(),))} \\
{ready_arguments} \\
  --host-output {quote_command((host_log,))} \\
  --solver-output {quote_command((solver_log,))} \\
  --readiness-timeout-ms {round(readiness_timeout * 1000)} \\
  --termination-grace-ms {round(termination_grace * 1000)} \\
  --host srun --nodes={scheduler.nodes} --ntasks={runtime['host_tasks']} \\
    --ntasks-per-node=1 \\
    --cpus-per-task={runtime['host_cpus_per_task']} --cpu-bind=none \\
{host_gres}    --exact --exclusive \\
    {quote_command(host)} \\
  --solver srun --nodes={scheduler.nodes} --ntasks={scheduler.ntasks} \\
{solver_mpi}    --ntasks-per-node={runtime['solver_tasks_per_node']} \\
    --cpus-per-task={scheduler.cpus_per_task} --cpu-bind=none \\
{solver_gres}    --exact --exclusive \\
    {quote_command(solver)}"""
    slurm = work_dir / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    batch = slurm / "longship.sbatch"
    batch.write_text(
        _render(
            "sailing.sbatch.in",
            {
                "JOB_NAME": job_name,
                "ACCOUNT_DIRECTIVE": account_directive,
                "PARTITION_DIRECTIVE": partition_directive,
                "TIME_LIMIT": scheduler.time,
                "NODES": scheduler.nodes,
                "ALLOCATION_TASKS": allocation_tasks,
                "ALLOCATION_CPUS_PER_TASK": allocation_cpus_per_task,
                "MEMORY_DIRECTIVE": memory,
                "GRES_DIRECTIVE": gres_directive,
                "MEMORY_SANITIZER": memory_sanitizer,
                "SAILING_LOG": longship_log,
                "LAUNCH_BODY": launch_body,
            },
        ),
        encoding="utf-8",
    )
    return batch


def write_submission_wrapper(work_dir: Path, batch: Path) -> Path:
    slurm = work_dir / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    wrapper = slurm / "submit.sh"
    wrapper.write_text(
        _render(
            "submit.sh.in",
            {
                "BATCH_PATH": quote_command((batch,)),
                "JOB_FILE": quote_command((internal_path(work_dir, "job.id"),)),
            },
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o750)
    return wrapper


def force_cancel(work_dir: Path) -> None:
    job_file = internal_path(work_dir, "job.id")
    identity_deadline = time.monotonic() + 5.0
    job_id = ""
    while time.monotonic() < identity_deadline:
        try:
            job_id = job_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        if job_id:
            break
        time.sleep(0.05)
    if not job_id:
        raise RuntimeError("Slurm job identity was not published within 5 seconds")
    subprocess.run(
        ["scancel", "--signal=KILL", job_id],
        check=True,
        timeout=5.0,
    )
