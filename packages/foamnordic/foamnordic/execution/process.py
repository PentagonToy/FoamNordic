"""Locate and launch native Longship processes."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from time import monotonic

from .run import Run
from .run_files import initialize_harbor_log, initialize_sailing_log, sailing_paths
from .runtime_paths import active_runtime_candidates


def longship_executable() -> Path:
    candidates: list[Path] = []
    override = os.environ.get("FOAMNORDIC_LONGSHIP")
    if override:
        candidates.append(Path(override).expanduser())

    candidates.extend(
        runtime / "bin/foamnordic-longship"
        for runtime in active_runtime_candidates()
    )

    candidates.append(
        Path(__file__).resolve().parents[1] / "bin/foamnordic-longship"
    )

    native_spec = find_spec("foamnordic._native")
    if native_spec is not None and native_spec.origin:
        candidates.append(
            Path(native_spec.origin).resolve().parent
            / "bin/foamnordic-longship"
        )

    rejected: list[Path] = []
    for executable in dict.fromkeys(candidates):
        if container_private_path(executable):
            rejected.append(executable)
            continue
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable.resolve()

    checked = ", ".join(str(path) for path in candidates)
    rejected_note = (
        " Container-private paths were rejected: "
        + ", ".join(str(path) for path in rejected)
        if rejected
        else ""
    )
    raise RuntimeError(
        "The Longship executable is unavailable. Install a binary wheel or "
        "an editable/Git build containing the native runtime. "
        f"Checked: {checked}.{rejected_note}"
    )


def container_private_path(path: Path) -> bool:
    """Reject known ephemeral container build roots from external launch plans."""

    return any(part.startswith("ROIHU_TYKKY_") for part in path.parts)


def launch_local(
    *,
    host: Sequence[str],
    solver: Sequence[str],
    ready_files: Sequence[Path],
    work_dir: Path,
    readiness_timeout: float = 30.0,
    termination_grace: float = 2.0,
    plan_digest: str | None = None,
    name: str = "foamnordic",
    observation_sources: int = 1,
    cleanup_paths: Sequence[Path] = (),
) -> Run:
    """Launch one local host-and-solver plan under the native supervisor."""

    if not host or not solver or not ready_files:
        raise ValueError("host, solver, and ready_files must not be empty")
    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    longship_log, host_log, solver_log = sailing_paths(work_dir, name)
    arguments = [str(longship_executable())]
    for ready in ready_files:
        arguments.extend(("--ready", str(ready)))
    arguments.extend(
        (
            "--host-output",
            str(host_log),
            "--solver-output",
            str(solver_log),
            "--readiness-timeout-ms",
            str(round(readiness_timeout * 1000)),
            "--termination-grace-ms",
            str(round(termination_grace * 1000)),
            "--host",
            *map(str, host),
            "--solver",
            *map(str, solver),
        )
    )
    return launch_process(
        arguments,
        work_dir=work_dir,
        process_log=longship_log,
        longship_log=longship_log,
        host_log=host_log,
        solver_log=solver_log,
        plan_digest=plan_digest,
        name=name,
        cleanup_paths=(*ready_files, *cleanup_paths),
        observation_sources=observation_sources,
    )


def launch_process(
    arguments: Sequence[str | Path],
    *,
    work_dir: Path,
    process_log: Path,
    longship_log: Path,
    host_log: Path,
    solver_log: Path,
    plan_digest: str | None = None,
    name: str = "foamnordic",
    job_file: Path | None = None,
    partition: str = "local",
    force_cancel: Callable[[], None] | None = None,
    cleanup_paths: Sequence[Path] = (),
    observation_sources: int = 1,
    environment: Mapping[str, str] | None = None,
    orphan_timeout: float | None = None,
) -> Run:
    """Start a lifecycle-owning process and bind it to the public Run handle."""

    started_at = datetime.now().astimezone()
    initialize_sailing_log(longship_log, name)
    initialize_harbor_log(host_log, name)
    mode = "ab" if process_log == longship_log else "wb"
    stream = process_log.open(mode)
    owner_reader: int | None = None
    owner_writer: int | None = None
    child_environment = None if environment is None else dict(environment)
    pass_fds: tuple[int, ...] = ()
    if orphan_timeout is not None:
        if os.name != "posix":
            stream.close()
            raise RuntimeError("owner-loss monitoring requires a POSIX platform")
        owner_reader, owner_writer = os.pipe()
        child_environment = (
            dict(os.environ) if child_environment is None else child_environment
        )
        child_environment["FOAMNORDIC_OWNER_FD"] = str(owner_reader)
        child_environment["FOAMNORDIC_ORPHAN_TIMEOUT"] = str(orphan_timeout)
        pass_fds = (owner_reader,)
    try:
        process = subprocess.Popen(
            [str(value) for value in arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_environment,
            pass_fds=pass_fds,
        )
    except Exception:
        stream.close()
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        for descriptor in (owner_reader, owner_writer):
            if descriptor is not None:
                os.close(descriptor)
        raise
    if owner_reader is not None:
        os.close(owner_reader)
    return Run(
        process,
        started=monotonic(),
        work_dir=work_dir,
        longship_log=longship_log,
        host_log=host_log,
        solver_log=solver_log,
        log_stream=stream,
        plan_digest=plan_digest,
        name=name,
        job_file=job_file,
        partition=partition,
        force_cancel=force_cancel,
        cleanup_paths=cleanup_paths,
        observation_sources=observation_sources,
        started_at=started_at,
        owner_descriptor=owner_writer,
    )

__all__ = [
    "container_private_path",
    "launch_local",
    "launch_process",
    "longship_executable",
]
