"""Slurm rendering, submission ownership, and force cancellation."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import re
import subprocess
import time
from typing import Mapping, Sequence, TYPE_CHECKING

from ._run import _banner, _internal_path, _longship_executable, _sailing_paths
from ._shell import quote_command

if TYPE_CHECKING:
    from ._spec import Longship


def _template(name: str) -> str:
    packaged = files("foamnordic").joinpath(f"templates/slurm/{name}")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).resolve().parents[2]
        / f"src/foamnordic/template/slurm/{name}"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError(f"FoamNordic Slurm template is unavailable: {name}")


def _render(name: str, variables: Mapping[str, object]) -> str:
    rendered = _template(name)
    for key, value in variables.items():
        rendered = rendered.replace(f"@{key}@", str(value))
    unresolved = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved Slurm template variables: {unresolved}")
    return rendered


def write_batch(
    longship: Longship,
    runtime: Mapping[str, object],
    work_dir: Path,
    host: Sequence[str] | None,
    solver: Sequence[str],
    ready: Path | None,
    readiness_timeout: float,
    termination_grace: float,
) -> Path:
    scheduler = longship.scheduler
    assert scheduler is not None
    job_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in longship.name
    )[:128]
    memory = (
        ""
        if scheduler.mem_per_cpu is None
        else (
            f"#SBATCH --mem-per-cpu={scheduler.mem_per_cpu}"
            "      # Memory reserved per CPU core\n"
        )
    )
    memory_sanitizer = (
        "unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU"
        if scheduler.mem_per_cpu is not None
        else "unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE"
    )
    longship_log, host_log, solver_log = _sailing_paths(work_dir, longship.name)
    banner = "\n".join(f"printf '%s\\n' {quote_command((line,))}" for line in _banner().splitlines())
    heading = (
        f"{banner}\n"
        f"printf '%s\\n' {quote_command((f'[FoamNordic] Sailing: {longship.name}',))}"
    )
    if host is None:
        launch_body = f""": > {quote_command((host_log,))}
exec srun \\
  --nodes={scheduler.nodes} \\
  --ntasks={scheduler.ntasks} \\
  --ntasks-per-node={runtime['solver_tasks_per_node']} \\
  --cpus-per-task={scheduler.cpus_per_task} \\
  --cpu-bind=none \\
  --output={quote_command((solver_log,))} \\
  --exact \\
  --exclusive \\
  {quote_command(solver)}"""
    else:
        assert ready is not None
        launch_body = f"""exec {quote_command((_longship_executable(),))} \\
  --ready {quote_command((ready,))} \\
  --host-output {quote_command((host_log,))} \\
  --solver-output {quote_command((solver_log,))} \\
  --readiness-timeout-ms {round(readiness_timeout * 1000)} \\
  --termination-grace-ms {round(termination_grace * 1000)} \\
  --host srun --nodes={scheduler.nodes} --ntasks={runtime['host_tasks']} \\
    --ntasks-per-node=1 \\
    --cpus-per-task={runtime['host_cpus_per_task']} --cpu-bind=none \\
    --exact --exclusive \\
    {quote_command(host)} \\
  --solver srun --nodes={scheduler.nodes} --ntasks={scheduler.ntasks} \\
    --ntasks-per-node={runtime['solver_tasks_per_node']} \\
    --cpus-per-task={scheduler.cpus_per_task} --cpu-bind=none \\
    --exact --exclusive \\
    {quote_command(solver)}"""
    slurm = work_dir / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    batch = slurm / "longship.sbatch"
    batch.write_text(
        _render(
            "sailing.sbatch.in",
            {
                "JOB_NAME": job_name,
                "ACCOUNT": scheduler.account,
                "PARTITION": scheduler.partition,
                "TIME_LIMIT": scheduler.time,
                "NODES": scheduler.nodes,
                "ALLOCATION_TASKS": scheduler.ntasks + int(runtime["host_tasks"]),
                "ALLOCATION_CPUS_PER_TASK": max(
                    scheduler.cpus_per_task,
                    int(runtime["host_cpus_per_task"]),
                ),
                "MEMORY_DIRECTIVE": memory,
                "MEMORY_SANITIZER": memory_sanitizer,
                "SAILING_LOG": longship_log,
                "BANNER": heading,
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
                "JOB_FILE": quote_command((_internal_path(work_dir, "job.id"),)),
            },
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o750)
    return wrapper


def force_cancel(work_dir: Path) -> None:
    job_file = _internal_path(work_dir, "job.id")
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
    subprocess.run(["scancel", "--signal=KILL", job_id], check=True)
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        active = subprocess.run(
            ["squeue", "--noheader", "--job", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if not active.stdout.strip():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Slurm job {job_id} remained active after force cancellation")
