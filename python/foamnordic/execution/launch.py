"""Compile declarations and connect case, runtime, and scheduler backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import TYPE_CHECKING

from .._case import PreparedProgram, prepare_case, validate_case
from .run import Run, _internal_path, _launch_local, _launch_process, _sailing_paths
from .shell import quote_command, toolchain_shell
from .slurm import force_cancel, write_batch, write_submission_wrapper
from .runtime_paths import active_runtime_candidates, toolchain_runtime_candidates

try:
    from .. import _native
except ImportError:
    _native = None

if TYPE_CHECKING:
    from ..core.spec import Longship


_INHERITED_SLURM_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_")


def _programs(longship: Longship):
    return longship.field_programs


def _submission_environment() -> dict[str, str]:
    """Copy the caller environment without a parent Slurm allocation."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_INHERITED_SLURM_PREFIXES)
    }


def _worker(longship: Longship | None = None) -> Path:
    explicit = os.environ.get("FOAMNORDIC_CLOSURE_WORKER")
    candidates = [] if explicit is None else [Path(explicit)]
    prepared = os.environ.get("FOAMNORDIC_PREPARED_WORK_DIR")
    if prepared:
        candidates.append(
            Path(prepared) / "build/tools/resident/foamnordic_closure_worker"
        )
    if longship is not None:
        candidates.extend(
            path / "bin/foamnordic_closure_worker"
            for path in toolchain_runtime_candidates(longship.case._toolchain)
        )
    candidates.extend(
        path / "bin/foamnordic_closure_worker"
        for path in active_runtime_candidates()
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


def _openfoam_library(longship: Longship | None = None) -> Path:
    explicit = os.environ.get("FOAMNORDIC_OPENFOAM_LIB")
    prepared = os.environ.get("FOAMNORDIC_PREPARED_WORK_DIR")
    candidates = [] if explicit is None else [Path(explicit)]
    if prepared:
        candidates.append(Path(prepared) / "lib")
    if longship is not None:
        candidates.extend(
            path / "lib"
            for path in toolchain_runtime_candidates(longship.case._toolchain)
        )
    candidates.extend(path / "lib" for path in active_runtime_candidates())
    candidates.append(Path.home() / ".local/share/foamnordic/native/lib")
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        libraries = (
            (path,)
            if path.is_file()
            else (
                *sorted(path.glob("libfoamnordicOpenFOAM-*.dylib")),
                path / "libfoamnordicOpenFOAM.so",
                path / "libfoamnordicOpenFOAM.dylib",
            )
        )
        for library in libraries:
            if library.is_file():
                return library
    raise RuntimeError(
        "OpenFOAM integration library is unavailable for the Case.of_cmd ABI. "
        "Load that OpenFOAM environment and run `foamnordic build`; "
        "FOAMNORDIC_OPENFOAM_LIB remains available as an explicit override."
    )


def _solver_command(
    longship: Longship,
    case_dir: Path,
    *,
    local_mpi: bool,
    openfoam_library: Path | None = None,
) -> tuple[str, ...]:
    environment = ""
    if _programs(longship):
        library = openfoam_library or _openfoam_library(longship)
        library_directory = library if library.is_dir() else library.parent
        application_directory = library_directory.parent / "bin"
        environment = (
            f"export FOAM_USER_LIBBIN={shlex.quote(str(library_directory))}; "
            f"export PATH={shlex.quote(str(application_directory))}:${{PATH}}; "
            f"export LD_LIBRARY_PATH={shlex.quote(str(library_directory))}:${{LD_LIBRARY_PATH:-}}; "
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


def _host_command(
    longship: Longship,
    prepared: PreparedProgram,
) -> tuple[str, ...]:
    program = prepared.program
    if program.artifact is None and prepared.artifact is None:
        raise RuntimeError("a model-backed field program is required")
    selected_artifact = prepared.artifact or program.artifact
    assert selected_artifact is not None
    artifact = selected_artifact.expanduser().resolve()
    metadata = _artifact_metadata(artifact)
    model_format = str(metadata["format"])
    manifest_inputs = tuple(str(item[0]) for item in metadata["inputs"])
    manifest_outputs = tuple(str(item[0]) for item in metadata["outputs"])
    if manifest_inputs != tuple(program.inputs):
        raise ValueError(
            "field-program input order does not match its FNOM manifest: "
            f"{tuple(program.inputs)!r} != {manifest_inputs!r}"
        )
    if manifest_outputs != tuple(program.outputs):
        raise ValueError(
            "field-program output order does not match its FNOM manifest: "
            f"{tuple(program.outputs)!r} != {manifest_outputs!r}"
        )
    executable: list[object]
    if model_format == "onnx":
        executable = [_worker(longship)]
    elif model_format in {"joblib", "equinox"}:
        executable = [sys.executable, "-m", "foamnordic.execution.resident"]
    else:
        raise ValueError(f"unsupported field-program artifact format: {model_format}")
    values: list[object] = [
        *executable,
        f"unix://{prepared.socket}",
        artifact,
        "--connections",
        str(longship.case.ranks),
        "--ready-file",
        prepared.ready,
    ]
    if model_format in {"joblib", "equinox"}:
        values.extend(("--key", program.key.to_json(), "--program", program.name))
    if longship.placement.data_path == "uds":
        values.append("--no-shm")
    # ClosureHost is a FoamNordic backend process, not an OpenFOAM
    # application.  In particular, wrapping it with OpenFOAM.app on macOS
    # leaves an unnecessary outer process alive after the worker exits.  The
    # solver alone owns the case toolchain; backend executables carry their
    # own runtime paths or use the active Python environment.
    return tuple(str(value) for value in values)


def _host_group_command(
    longship: Longship,
    prepared: tuple[PreparedProgram, ...],
    work_dir: Path,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    commands = [_host_command(longship, item) for item in prepared]
    if len(commands) == 1:
        return commands[0], (prepared[0].ready,)
    aggregate = _internal_path(work_dir, "programs.ready")
    configuration = _internal_path(work_dir, "host-programs.json")
    configuration.write_text(
        json.dumps(
            {
                "commands": [list(command) for command in commands],
                "ready_files": [str(item.ready) for item in prepared],
                "aggregate_ready": str(aggregate),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return (sys.executable, "-m", "foamnordic.execution.host_group", str(configuration)), (
        aggregate,
    )


def launch(
    longship: Longship,
    *,
    readiness_timeout: float = 120.0,
    termination_grace: float = 30.0,
    verbose: bool = True,
) -> Run:
    """Compile, prepare, and start one non-blocking coupled workload."""

    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    validate_case(longship)
    plan = longship.compile()
    integration_library = (
        _openfoam_library(longship) if _programs(longship) else None
    )
    work_dir, case_dir, prepared = prepare_case(
        longship,
        plan,
        integration_library,
        verbose=verbose,
    )
    local = longship.scheduler is None
    solver = _solver_command(
        longship,
        case_dir,
        local_mpi=local,
        openfoam_library=integration_library,
    )
    host, ready_files = (
        _host_group_command(longship, prepared, work_dir)
        if prepared
        else (None, ())
    )

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
            ready_files=ready_files,
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
        ready_files[0] if host is not None else None,
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
