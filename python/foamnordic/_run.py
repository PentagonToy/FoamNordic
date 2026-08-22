"""Asynchronous Python ownership of a native Longship process."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
from time import monotonic, sleep
from typing import Callable, Mapping, Sequence, TextIO


_ACTIVE_RUNS: set[Run] = set()


def _shutdown_active_runs() -> None:
    for run in tuple(_ACTIVE_RUNS):
        run._shutdown_owned_process()


atexit.register(_shutdown_active_runs)


def _safe_name(name: str) -> str:
    value = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-")
    return value[:128] or "FoamNordic"


def _sailing_paths(work_dir: Path, name: str) -> tuple[Path, Path, Path]:
    logs = Path(work_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(name)
    return (
        logs / f"Sailing_{stem}.log",
        logs / f"Harbor_{stem}.log",
        logs / f"Sailing_{stem}.out",
    )


def _internal_path(work_dir: Path, name: str) -> Path:
    internal = Path(work_dir) / ".foamnordic"
    internal.mkdir(parents=True, exist_ok=True)
    return internal / name


def _banner() -> str:
    packaged = files("foamnordic").joinpath("templates/large_banner.txt")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8").rstrip()
    source = (
        Path(__file__).resolve().parents[2]
        / "src/foamnordic/template/large_banner.txt"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8").rstrip()
    return "FoamNordic"


def _initialize_sailing_log(path: Path, name: str) -> None:
    path.write_text(
        f"{_banner()}\n\n[FoamNordic] Sailing: {name}\n",
        encoding="utf-8",
    )


class RunStatus(str, Enum):
    """Observable lifecycle states of a launched Longship workload."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Result:
    """Durable terminal state and log locations for one Longship run."""

    status: RunStatus
    exit_code: int
    elapsed_seconds: float
    work_dir: Path
    longship_log: Path
    host_log: Path
    solver_log: Path
    plan_digest: str | None = None
    name: str = "foamnordic"
    job_id: str | None = None
    partition: str = "local"
    node: str = "-"

    @property
    def success(self) -> bool:
        """Return whether both coupled components completed successfully."""

        return self.status is RunStatus.SUCCEEDED

    def summary(
        self,
        style: str = "compact",
        *,
        display: bool = True,
    ) -> RunSummary:
        """Return and optionally display a compact terminal summary."""

        normalized = _summary_style(style)
        details = _query_slurm(self.job_id) if self.job_id is not None else {}
        summary = RunSummary(
            job_id=self.job_id or "-",
            name=self.name,
            status=details.get("status", self.status.value),
            partition=details.get("partition", self.partition),
            node=details.get("node", self.node),
            elapsed=details.get("elapsed", _format_elapsed(self.elapsed_seconds)),
            style=normalized,
            exit_code=self.exit_code,
            work_dir=self.work_dir,
            sailing_log=self.longship_log,
            sailing_output=self.solver_log,
            harbor_log=self.host_log,
            plan_digest=self.plan_digest,
        )
        if display:
            summary.display()
        return summary


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Compact scheduler-neutral identity and state for one workload."""

    job_id: str
    name: str
    status: str
    partition: str
    node: str
    elapsed: str
    style: str = "compact"
    exit_code: int | None = None
    work_dir: Path | None = None
    sailing_log: Path | None = None
    sailing_output: Path | None = None
    harbor_log: Path | None = None
    plan_digest: str | None = None

    def display(self) -> None:
        import onsaemiro as osm

        if self.style == "expanded":
            table = osm.TableMaker(
                title=f"{self.name} Result",
                columns=["Property", "Value"],
                mode="static",
            )
            rows = (
                ("Job ID", self.job_id),
                ("Name", self.name),
                ("Status", self.status),
                ("Exit code", "-" if self.exit_code is None else self.exit_code),
                ("Partition", self.partition),
                ("Node", self.node),
                ("Elapsed", self.elapsed),
                ("Work directory", self.work_dir or "-"),
                ("Sailing log", self.sailing_log or "-"),
                ("OpenFOAM output", self.sailing_output or "-"),
                ("Harbor log", self.harbor_log or "-"),
                ("Plan digest", self.plan_digest or "-"),
            )
            for row in rows:
                table.add_row(row)
            table.display()
            return
        table = osm.TableMaker(
            title=f"{self.name} Summary",
            columns=["Job ID", "Name", "Status", "Partition", "Node", "Elapsed"],
            mode="static",
        )
        table.add_row(
            self.job_id,
            self.name,
            self.status,
            self.partition,
            self.node,
            self.elapsed,
        )
        table.display()


def _summary_style(value: str) -> str:
    styles = {
        "short": "compact",
        "compact": "compact",
        "long": "expanded",
        "expanded": "expanded",
    }
    try:
        return styles[value]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "summary style must be short, compact, long, or expanded"
        ) from error


def _format_elapsed(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Run:
    """A non-blocking native Longship process with one terminal stop operation."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        started: float,
        work_dir: Path,
        longship_log: Path,
        host_log: Path,
        solver_log: Path,
        log_stream: TextIO,
        plan_digest: str | None = None,
        name: str = "foamnordic",
        job_file: Path | None = None,
        partition: str = "local",
        force_cancel: Callable[[], None] | None = None,
        cleanup_paths: Sequence[Path] = (),
        observation_sources: int = 1,
    ) -> None:
        self._process = process
        self._started = started
        self._work_dir = work_dir
        self._longship_log = longship_log
        self._host_log = host_log
        self._solver_log = solver_log
        self._log_stream = log_stream
        self._plan_digest = plan_digest
        self._name = name
        self._job_file = job_file
        self._partition = partition
        self._force_cancel = force_cancel
        self._cleanup_paths = tuple(Path(path) for path in cleanup_paths)
        self._observation_sources = observation_sources
        self._cancel_requested = False
        self._detached = False
        self._result: Result | None = None
        self._lock = threading.Lock()
        _ACTIVE_RUNS.add(self)

    @property
    def pid(self) -> int:
        """Operating-system process identifier of the Longship supervisor."""

        return self._process.pid

    @property
    def status(self) -> RunStatus:
        """Return current state without blocking."""

        with self._lock:
            if self._result is not None:
                return self._result.status
            exit_code = self._process.poll()
            if exit_code is None:
                return RunStatus.RUNNING
            return self._finish_locked(exit_code).status

    def _wait(self, timeout: float | None = None) -> Result:
        """Wait for terminal state and return the same immutable result each time."""

        with self._lock:
            if self._result is not None:
                return self._result
        try:
            exit_code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Longship is still running") from error
        with self._lock:
            return self._finish_locked(exit_code)

    def stop(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> Result:
        """Wait normally, or force-stop every process owned by this run."""

        if timeout is not None and timeout <= 0:
            raise ValueError("stop timeout must be positive")
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        if not force:
            return self._wait(timeout=timeout)
        with self._lock:
            if self._result is not None:
                return self._result
            running = self._process.poll() is None
            self._cancel_requested = running
        if running:
            if self._force_cancel is not None:
                self._force_cancel()
            else:
                # Longship converts TERM into bounded child-group teardown,
                # including SIGKILL for a component that ignores its grace.
                self._process.terminate()
        try:
            result = self._wait(timeout=timeout or 90.0)
        except TimeoutError as error:
            raise TimeoutError(
                "the forced workload termination did not reach a terminal state"
            ) from error
        for path in self._cleanup_paths:
            path.unlink(missing_ok=True)
        return result

    def detach(self) -> "Run":
        """Allow this workload to outlive an orderly Python kernel shutdown."""

        with self._lock:
            self._detached = True
        _ACTIVE_RUNS.discard(self)
        return self

    def _wait_for_start(
        self,
        timeout: float | None,
    ) -> tuple[str | None, str]:
        """Wait for Slurm to start or terminate the submitted workload."""

        deadline = None if timeout is None else monotonic() + timeout
        job_id: str | None = None
        state = "submitting"
        while True:
            job_id = job_id or self._read_job_id()
            if job_id is not None:
                state = _query_slurm(job_id).get("status", state)
                if state in {
                    RunStatus.RUNNING.value,
                    RunStatus.SUCCEEDED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                }:
                    return job_id, state
            if self._process.poll() is not None:
                return job_id, self.status.value
            if deadline is not None and monotonic() >= deadline:
                return job_id, state
            sleep(0.25)

    def observe(self, *, poll_interval: float = 0.1):
        """Return an iterable observation stream; no context manager is required."""

        from ._observe import ObservationStream

        return ObservationStream(
            self,
            self._work_dir / "observations/observations.jsonl",
            poll_interval=poll_interval,
            expected_sources=self._observation_sources,
        )

    def summary(
        self,
        style: str = "compact",
        *,
        display: bool = True,
    ) -> RunSummary:
        """Return and optionally display current local or Slurm metadata."""

        normalized = _summary_style(style)
        job_id = self._read_job_id()
        partition = self._partition
        node = socket.gethostname()
        elapsed = _format_elapsed(monotonic() - self._started)
        status = self.status.value
        if job_id is not None:
            details = _query_slurm(job_id)
            partition = details.get("partition", partition)
            node = details.get("node", "Slurm")
            elapsed = details.get("elapsed", elapsed)
            status = details.get("status", status)
        summary = RunSummary(
            job_id=job_id or f"local:{self.pid}",
            name=self._name,
            status=status,
            partition=partition,
            node=node,
            elapsed=elapsed,
            style=normalized,
            work_dir=self._work_dir,
            sailing_log=self._longship_log,
            sailing_output=self._solver_log,
            harbor_log=self._host_log,
            plan_digest=self._plan_digest,
        )
        if display:
            summary.display()
        return summary

    def _read_job_id(self) -> str | None:
        if self._job_file is None:
            return None
        try:
            value = self._job_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def _finish_locked(self, exit_code: int) -> Result:
        if self._result is not None:
            return self._result
        self._log_stream.close()
        cancelled = self._cancel_requested or exit_code == 130
        status = (
            RunStatus.CANCELLED
            if cancelled
            else RunStatus.SUCCEEDED
            if exit_code == 0
            else RunStatus.FAILED
        )
        with self._longship_log.open("a", encoding="utf-8") as stream:
            print(
                f"[FoamNordic] Sailing completed: {status.value} ({exit_code})",
                file=stream,
            )
        self._finalize_log_names()
        self._result = Result(
            status=status,
            exit_code=exit_code,
            elapsed_seconds=monotonic() - self._started,
            work_dir=self._work_dir,
            longship_log=self._longship_log,
            host_log=self._host_log,
            solver_log=self._solver_log,
            plan_digest=self._plan_digest,
            name=self._name,
            job_id=self._read_job_id(),
            partition=self._partition,
            node=socket.gethostname(),
        )
        _ACTIVE_RUNS.discard(self)
        return self._result

    def _shutdown_owned_process(self) -> None:
        """Best-effort bounded teardown for an orderly interpreter shutdown."""

        with self._lock:
            if self._detached or self._result is not None:
                return
            if self._process.poll() is not None:
                return
            self._cancel_requested = True
        job_id = self._read_job_id()
        if job_id is not None and shutil.which("scancel") is not None:
            try:
                subprocess.run(
                    ["scancel", "--signal=KILL", job_id],
                    check=False,
                    timeout=5.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            self._process.terminate()
        except OSError:
            pass

    def _finalize_log_names(self) -> None:
        identity = self._read_job_id() or f"local-{self.pid}"
        suffix = _safe_name(identity)
        renamed: list[Path] = []
        for path in (self._longship_log, self._host_log, self._solver_log):
            destination = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            if path.exists() and path != destination:
                path.replace(destination)
            renamed.append(destination)
        self._longship_log, self._host_log, self._solver_log = renamed


def _query_slurm(job_id: str) -> dict[str, str]:
    sacct = shutil.which("sacct")
    if sacct is None:
        return {}
    try:
        result = subprocess.run(
            [
                sacct,
                "--jobs",
                job_id,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,Partition,Elapsed,NodeList",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    records = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    selected = next(
        (record for record in records if record and record[0] == job_id),
        records[0] if records else None,
    )
    if selected is None or len(selected) < 5:
        return {}
    state = selected[1].split("+", 1)[0].strip().split(" ", 1)[0].lower()
    status = _normalize_slurm_state(state)
    return {
        "status": status,
        "partition": selected[2] or "-",
        "elapsed": selected[3] or "-",
        "node": selected[4] or "Slurm",
    }


def _normalize_slurm_state(state: str) -> str:
    if state in {"pending", "configuring", "requeued", "resizing"}:
        return "pending"
    if state in {"running", "completing", "stage_out"}:
        return RunStatus.RUNNING.value
    if state == "completed":
        return RunStatus.SUCCEEDED.value
    if state == "cancelled":
        return RunStatus.CANCELLED.value
    return RunStatus.FAILED.value if state else "unknown"


def _longship_executable() -> Path:
    executable = Path(__file__).with_name("bin") / "foamnordic-longship"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            "The packaged Longship executable is unavailable; install a binary wheel"
        )
    return executable


def _launch_local(
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
) -> Run:
    """Internal lifecycle primitive used by the future plan renderer."""

    if not host or not solver or not ready_files:
        raise ValueError("host, solver, and ready_files must not be empty")
    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    longship_log, host_log, solver_log = _sailing_paths(work_dir, name)
    arguments = [str(_longship_executable())]
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
    return _launch_process(
        arguments,
        work_dir=work_dir,
        process_log=longship_log,
        longship_log=longship_log,
        host_log=host_log,
        solver_log=solver_log,
        plan_digest=plan_digest,
        name=name,
        cleanup_paths=ready_files,
        observation_sources=observation_sources,
    )


def _launch_process(
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
) -> Run:
    """Start a lifecycle-owning process and bind it to the public Run handle."""

    _initialize_sailing_log(longship_log, name)
    mode = "ab" if process_log == longship_log else "wb"
    stream = process_log.open(mode)
    try:
        process = subprocess.Popen(
            [str(value) for value in arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=None if environment is None else dict(environment),
        )
    except Exception:
        stream.close()
        raise
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
    )
