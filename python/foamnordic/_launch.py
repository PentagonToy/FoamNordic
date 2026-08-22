"""Compile declarations and connect case, runtime, and scheduler backends."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import TYPE_CHECKING

from ._case import prepare_case, validate_case
from ._run import Run, _internal_path, _launch_local, _launch_process, _sailing_paths
from ._shell import quote_command, toolchain_shell
from ._slurm import force_cancel, write_batch, write_submission_wrapper

try:
    from . import _native
except ImportError:
    _native = None

if TYPE_CHECKING:
    from ._spec import Longship


_INHERITED_SLURM_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_")


def _submission_environment() -> dict[str, str]:
    """Copy the caller environment without a parent Slurm allocation."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_INHERITED_SLURM_PREFIXES)
    }


def _worker() -> Path:
    explicit = os.environ.get("FOAMNORDIC_CLOSURE_WORKER")
    candidates = [] if explicit is None else [Path(explicit)]
    prepared = os.environ.get("FOAMNORDIC_PREPARED_WORK_DIR")
    if prepared:
        candidates.append(
            Path(prepared) / "build/tools/resident/foamnordic_closure_worker"
        )
    candidates.append(
        Path.home() / ".local/share/foamnordic/native/bin/foamnordic_closure_worker"
    )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "ClosureHost executable is unavailable. Set FOAMNORDIC_CLOSURE_WORKER "
        "or select a prepared OpenFOAM runtime profile."
    )


def _openfoam_library() -> Path:
    explicit = os.environ.get("FOAMNORDIC_OPENFOAM_LIB")
    prepared = os.environ.get("FOAMNORDIC_PREPARED_WORK_DIR")
    candidates = [] if explicit is None else [Path(explicit)]
    if prepared:
        candidates.append(Path(prepared) / "lib")
    candidates.append(Path.home() / ".local/share/foamnordic/native/lib")
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        library = path if path.is_file() else path / "libfoamnordicOpenFOAM.so"
        if library.is_file():
            return library.parent
    raise RuntimeError(
        "OpenFOAM integration library is unavailable. Set "
        "FOAMNORDIC_OPENFOAM_LIB to its directory or library file."
    )


def _solver_command(
    longship: Longship,
    case_dir: Path,
    *,
    local_mpi: bool,
) -> tuple[str, ...]:
    environment = ""
    if longship.closures:
        library = _openfoam_library()
        environment = (
            f"export FOAM_USER_LIBBIN={shlex.quote(str(library))}; "
            f"export LD_LIBRARY_PATH={shlex.quote(str(library))}:${{LD_LIBRARY_PATH:-}}; "
        )
    solver: list[object] = [longship.case.application, "-case", case_dir]
    if longship.case.ranks > 1:
        solver.append("-parallel")
    if local_mpi and longship.case.ranks > 1:
        solver = ["mpirun", "-np", str(longship.case.ranks), *solver]
    return toolchain_shell(
        longship.case._toolchain,
        environment + "exec " + quote_command(solver),
    )


def _artifact_metadata(path: Path) -> dict[str, object]:
    if _native is None:
        raise RuntimeError("model launch requires a FoamNordic binary wheel")
    return dict(_native.read_model_manifest(str(path.expanduser().resolve())))


def _host_command(longship: Longship, ready: Path) -> tuple[str, ...]:
    closure = longship.closures[0]
    artifact = closure.artifact.expanduser().resolve()
    metadata = _artifact_metadata(artifact)
    model_format = str(metadata["format"])
    manifest_inputs = tuple(str(item[0]) for item in metadata["inputs"])
    manifest_outputs = tuple(str(item[0]) for item in metadata["outputs"])
    if manifest_inputs != tuple(closure.inputs):
        raise ValueError(
            "closure input order does not match its FNOM manifest: "
            f"{tuple(closure.inputs)!r} != {manifest_inputs!r}"
        )
    if manifest_outputs != tuple(closure.outputs):
        raise ValueError(
            "closure output order does not match its FNOM manifest: "
            f"{tuple(closure.outputs)!r} != {manifest_outputs!r}"
        )
    executable: list[object]
    if model_format == "onnx":
        executable = [_worker()]
    elif model_format in {"joblib", "equinox"}:
        executable = [sys.executable, "-m", "foamnordic._resident"]
    else:
        raise ValueError(f"unsupported closure artifact format: {model_format}")
    values: list[object] = [
        *executable,
        f"unix://{ready.parent / 'closure.sock'}",
        artifact,
        "--connections",
        str(longship.case.ranks),
        "--ready-file",
        ready,
    ]
    if longship.placement.data_path == "uds":
        values.append("--no-shm")
    return toolchain_shell(
        longship.case._toolchain,
        "exec " + quote_command(values),
    )


def launch(
    longship: Longship,
    *,
    readiness_timeout: float = 120.0,
    termination_grace: float = 30.0,
) -> Run:
    """Compile, prepare, and start one non-blocking coupled workload."""

    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    validate_case(longship)
    plan = longship.compile()
    work_dir, case_dir, ready = prepare_case(longship, plan)
    local = longship.scheduler is None
    solver = _solver_command(longship, case_dir, local_mpi=local)
    host = _host_command(longship, ready) if longship.closures else None

    if local and host is None:
        longship_log, host_log, solver_log = _sailing_paths(work_dir, longship.name)
        host_log.touch()
        return _launch_process(
            solver,
            work_dir=work_dir,
            process_log=solver_log,
            longship_log=longship_log,
            host_log=host_log,
            solver_log=solver_log,
            plan_digest=plan.digest,
            name=longship.name,
            observation_sources=longship.case.ranks,
        )
    if local:
        assert host is not None
        return _launch_local(
            host=host,
            solver=solver,
            ready_files=(ready,),
            work_dir=work_dir,
            readiness_timeout=readiness_timeout,
            termination_grace=termination_grace,
            plan_digest=plan.digest,
            name=longship.name,
            observation_sources=longship.case.ranks,
        )

    for command in ("sbatch", "squeue", "sacct", "scancel"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Slurm command is unavailable: {command}")
    runtime = plan.as_dict()["runtime"]
    batch = write_batch(
        longship,
        runtime,
        work_dir,
        host,
        solver,
        ready if host is not None else None,
        readiness_timeout,
        termination_grace,
    )
    wrapper = write_submission_wrapper(work_dir, batch)
    longship_log, host_log, solver_log = _sailing_paths(work_dir, longship.name)
    assert longship.scheduler is not None
    return _launch_process(
        (wrapper,),
        work_dir=work_dir,
        process_log=_internal_path(work_dir, "submission.log"),
        longship_log=longship_log,
        host_log=host_log,
        solver_log=solver_log,
        plan_digest=plan.digest,
        name=longship.name,
        job_file=_internal_path(work_dir, "job.id"),
        partition=longship.scheduler.partition,
        force_cancel=lambda: force_cancel(work_dir),
        observation_sources=longship.case.ranks,
        environment=_submission_environment(),
    )
