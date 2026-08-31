"""Asynchronous Python ownership of a native Longship process."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from importlib.resources import files
from importlib.util import find_spec
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
from time import monotonic, sleep
from typing import Callable, Mapping, Sequence, TextIO

from ..core.managed import generated_kind, relocate_generated


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
        Path(__file__).resolve().parents[3]
        / "src/foamnordic/template/large_banner.txt"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8").rstrip()
    return "FoamNordic"


def _initialize_sailing_log(path: Path, name: str) -> None:
    if path.is_file():
        return
    path.write_text(
        f"{_banner()}\n\n[FoamNordic] Sailing: {name}\n",
        encoding="utf-8",
    )


def _initialize_harbor_log(path: Path, name: str) -> None:
    path.write_text(
        f"{_banner()}\n\n[FoamNordic] Harbor: {name}\n",
        encoding="utf-8",
    )


class RunStatus(str, Enum):
    """Observable lifecycle states of a launched Longship workload."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ResultArtifacts:
    """Stable locations owned by one generated run directory."""

    root: Path
    case: Path
    logs: Path
    observations: Path
    slurm: Path

    @property
    def existing(self) -> tuple[Path, ...]:
        """Return artifact locations that exist for this particular run."""

        return tuple(
            path
            for path in (
                self.case,
                self.logs,
                self.observations,
                self.slurm,
            )
            if path.exists()
        )


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

    @property
    def case(self) -> Path:
        """Return the isolated OpenFOAM case written by this run."""

        return self.work_dir / "case"

    @property
    def logs(self) -> Path:
        """Return the directory containing Sailing and Harbor logs."""

        return self.work_dir / "logs"

    @property
    def artifacts(self) -> ResultArtifacts:
        """Return stable paths without copying durable solver output."""

        return ResultArtifacts(
            root=self.work_dir,
            case=self.case,
            logs=self.logs,
            observations=self.work_dir / "observations",
            slurm=self.work_dir / "slurm",
        )

    @property
    def postprocess(self):
        """Open this result through the durable postprocessing API."""

        from ..postprocess import Case

        return Case(self)

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


def _slurm_elapsed_seconds(value: str) -> float | None:
    """Parse Slurm's ``[days-]hours:minutes:seconds`` representation."""

    match = re.fullmatch(
        r"(?:(?P<days>\d+)-)?(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)",
        value.strip(),
    )
    if match is None:
        return None
    return (
        int(match.group("days") or 0) * 86400
        + int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )


def _openfoam_clock_seconds(path: Path) -> float | None:
    """Read the final OpenFOAM ClockTime without loading a large log."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(
        r"ExecutionTime\s*=\s*[0-9.eE+-]+\s*s\s+ClockTime\s*=\s*"
        r"([0-9.eE+-]+)\s*s",
        text,
    )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


class _OpenFOAMProgress:
    """Incrementally render the latest OpenFOAM time without writing a log."""

    _TIME = re.compile(r"^\s*Time\s*=\s*([^\s]+)\s*$")
    _ITERATION = re.compile(r"^\s*Iteration\s*=\s*([^\s]+)\s*$")

    def __init__(self, path: Path, *, started: float, stream: TextIO) -> None:
        self._path = path
        self._started = started
        self._stream = stream
        self._offset = 0
        self._pending = ""
        self._latest: tuple[str, str] | None = None
        self._width = 0
        self._rendered = False

    def refresh(self, *, final: bool = False) -> None:
        try:
            size = self._path.stat().st_size
            if size < self._offset:
                self._offset = 0
                self._pending = ""
            with self._path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
        except OSError:
            return

        text = self._pending + chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        self._pending = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._pending = lines.pop()
        if final and self._pending:
            lines.append(self._pending)
            self._pending = ""

        for line in lines:
            match = self._TIME.match(line.rstrip("\r\n"))
            if match is not None:
                self._latest = ("t", match.group(1))
                continue
            match = self._ITERATION.match(line.rstrip("\r\n"))
            if match is not None:
                self._latest = ("iteration", match.group(1))

        if self._latest is None:
            return
        label, value = self._latest
        elapsed = _format_elapsed(monotonic() - self._started)
        message = (
            f"[FoamNordic] Sailing in OpenFOAM: {label} = {value}"
            f" | elapsed {elapsed}"
        )
        padding = " " * max(0, self._width - len(message))
        print(f"\r{message}{padding}", end="", flush=True, file=self._stream)
        self._width = len(message)
        self._rendered = True

    def clear(self) -> None:
        if not self._rendered:
            return
        print(f"\r{' ' * self._width}\r", end="", flush=True, file=self._stream)


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
        started_at: datetime | None = None,
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
        self._started_at = started_at or datetime.now().astimezone()
        identity_source = (
            f"{self._started_at.isoformat()}:{process.pid}:{work_dir}"
        ).encode("utf-8")
        identity_hash = hashlib.sha256(identity_source).hexdigest()[:6]
        self._local_identity = (
            f"local-{self._started_at:%Y%m%dT%H%M%S}-{identity_hash}"
        )
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

    def _wait(
        self,
        timeout: float | None = None,
        *,
        progress: bool = False,
    ) -> Result:
        """Wait for terminal state and return the same immutable result each time."""

        with self._lock:
            if self._result is not None:
                return self._result
        if not progress:
            try:
                exit_code = self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("Longship is still running") from error
            with self._lock:
                return self._finish_locked(exit_code)

        monitor = _OpenFOAMProgress(
            self._solver_log,
            started=self._started,
            stream=sys.stdout,
        )
        deadline = None if timeout is None else monotonic() + timeout
        try:
            while True:
                wait_timeout = 1.0
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Longship is still running")
                    wait_timeout = min(wait_timeout, remaining)
                try:
                    exit_code = self._process.wait(timeout=wait_timeout)
                    monitor.refresh(final=True)
                    break
                except subprocess.TimeoutExpired:
                    monitor.refresh()
        finally:
            monitor.clear()
        with self._lock:
            return self._finish_locked(exit_code)

    def stop(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
        progress: bool = False,
    ) -> Result:
        """Wait normally, optionally showing time, or force-stop owned processes."""

        if timeout is not None and timeout <= 0:
            raise ValueError("stop timeout must be positive")
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean")
        if not force:
            return self._wait(timeout=timeout, progress=progress)
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
        pending_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[str | None, str]:
        """Wait for Slurm to start or terminate the submitted workload."""

        deadline = None if timeout is None else monotonic() + timeout
        job_id: str | None = None
        state = "submitting"
        reported_pending: str | None = None
        while True:
            job_id = job_id or self._read_job_id()
            if job_id is not None:
                state = _query_slurm(job_id).get("status", state)
                if (
                    state == "pending"
                    and pending_callback is not None
                    and reported_pending != job_id
                ):
                    pending_callback(job_id, _query_slurm_estimated_start(job_id))
                    reported_pending = job_id
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

    def _slurm_start_time(self, job_id: str, *, estimated: bool = False) -> str:
        """Return Slurm's actual or estimated start timestamp when available."""

        if not estimated:
            actual = _query_slurm(job_id).get("start", "")
            if actual and actual not in {"Unknown", "N/A", "None"}:
                return actual
        return _query_slurm_estimated_start(job_id)

    def observe(
        self,
        *,
        poll_interval: float = 0.1,
        progress: bool = False,
    ):
        """Return observations, optionally rendering their latest solver time."""

        from .observe import ObservationStream

        return ObservationStream(
            self,
            self._work_dir / "observations/observations.jsonl",
            poll_interval=poll_interval,
            expected_sources=self._observation_sources,
            progress=progress,
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
        job_id = self._read_job_id()
        finished_at = datetime.now().astimezone()
        timing_started_at = self._started_at
        total_seconds = monotonic() - self._started
        if job_id is not None:
            details = _query_slurm(job_id)
            scheduled = _slurm_elapsed_seconds(details.get("elapsed", ""))
            if scheduled is not None:
                total_seconds = scheduled
            actual_start = details.get("start", "")
            if actual_start:
                try:
                    timing_started_at = datetime.fromisoformat(actual_start)
                    if timing_started_at.tzinfo is None:
                        timing_started_at = timing_started_at.replace(
                            tzinfo=self._started_at.tzinfo
                        )
                except ValueError:
                    pass
        openfoam_seconds = _openfoam_clock_seconds(self._solver_log)
        timing = (
            f"[FoamNordic] Timing: {_timestamp(timing_started_at)} -> "
            f"{_timestamp(finished_at)} | total={_format_elapsed(total_seconds)}"
        )
        if openfoam_seconds is not None:
            orchestration_seconds = max(0.0, total_seconds - openfoam_seconds)
            timing += (
                f" | OpenFOAM={_format_elapsed(openfoam_seconds)}"
                f" | orchestration={_format_elapsed(orchestration_seconds)}"
            )
        with self._longship_log.open("a", encoding="utf-8") as stream:
            print(
                f"[FoamNordic] Sailing completed: {status.value} ({exit_code})",
                file=stream,
            )
            print(timing, file=stream)
        self._finalize_log_names(job_id)
        self._finalize_work_directory(job_id)
        self._result = Result(
            status=status,
            exit_code=exit_code,
            elapsed_seconds=total_seconds,
            work_dir=self._work_dir,
            longship_log=self._longship_log,
            host_log=self._host_log,
            solver_log=self._solver_log,
            plan_digest=self._plan_digest,
            name=self._name,
            job_id=job_id,
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

    def _finalize_log_names(self, job_id: str | None = None) -> None:
        identity = job_id or self._local_identity
        suffix = _safe_name(identity)
        renamed: list[Path] = []
        for path in (self._longship_log, self._host_log, self._solver_log):
            destination = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            if path.exists() and path != destination:
                path.replace(destination)
            renamed.append(destination)
        self._longship_log, self._host_log, self._solver_log = renamed

    def _finalize_work_directory(self, job_id: str | None = None) -> None:
        # Low-level lifecycle primitives may be given an arbitrary directory.
        # Only a prepared, ownership-marked run is safe to rename here.
        if generated_kind(self._work_dir) != "run":
            return
        identity = f"slurm-{job_id}" if job_id is not None else self._local_identity
        name = _safe_name(self._name).replace("_", "-")
        destination = self._work_dir.parent / f"{name}-{identity}"
        if destination == self._work_dir:
            return
        if destination.exists():
            raise FileExistsError(
                f"FoamNordic final run directory already exists: {destination}"
            )
        previous = self._work_dir
        previous.replace(destination)
        relocate_generated(destination, previous=previous)

        def relocated(path: Path | None) -> Path | None:
            if path is None:
                return None
            try:
                relative = path.relative_to(previous)
            except ValueError:
                return path
            return destination / relative

        def relocated_required(path: Path) -> Path:
            value = relocated(path)
            assert value is not None
            return value

        self._work_dir = destination
        self._longship_log = relocated_required(self._longship_log)
        self._host_log = relocated_required(self._host_log)
        self._solver_log = relocated_required(self._solver_log)
        self._job_file = relocated(self._job_file)
        self._cleanup_paths = tuple(
            relocated(path) or path for path in self._cleanup_paths
        )


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
                "--format=JobIDRaw,State,Partition,Elapsed,NodeList,Start",
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
    if selected is None or len(selected) < 6:
        return {}
    state = selected[1].split("+", 1)[0].strip().split(" ", 1)[0].lower()
    status = _normalize_slurm_state(state)
    return {
        "status": status,
        "partition": selected[2] or "-",
        "elapsed": selected[3] or "-",
        "node": selected[4] or "Slurm",
        "start": selected[5] or "",
    }


def _query_slurm_estimated_start(job_id: str) -> str:
    squeue = shutil.which("squeue")
    if squeue is not None:
        try:
            result = subprocess.run(
                [
                    squeue,
                    "--start",
                    "--noheader",
                    "--job",
                    job_id,
                    "--format=%S",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            pass
        else:
            values = result.stdout.strip().splitlines()
            if values:
                selected = values[0].strip()
                if selected not in {"", "N/A", "Unknown", "None", "(null)"}:
                    return selected

    scontrol = shutil.which("scontrol")
    if scontrol is None:
        return ""
    try:
        result = subprocess.run(
            [scontrol, "show", "job", "--oneliner", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    match = re.search(r"(?:^|\s)StartTime=(\S+)", result.stdout)
    if match is None:
        return ""
    selected = match.group(1).strip()
    return "" if selected in {"N/A", "Unknown", "None", "(null)"} else selected


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
    candidates: list[Path] = []
    override = os.environ.get("FOAMNORDIC_LONGSHIP")
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(
        Path(__file__).resolve().parents[1] / "bin/foamnordic-longship"
    )

    native_spec = find_spec("foamnordic._native")
    if native_spec is not None and native_spec.origin:
        candidates.append(
            Path(native_spec.origin).resolve().parent
            / "bin/foamnordic-longship"
        )

    for executable in dict.fromkeys(candidates):
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable

    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "The Longship executable is unavailable. Install a binary wheel or "
        "an editable/Git build containing the native runtime. "
        f"Checked: {checked}"
    )


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

    started_at = datetime.now().astimezone()
    _initialize_sailing_log(longship_log, name)
    _initialize_harbor_log(host_log, name)
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
        started_at=started_at,
    )
