"""Compile declarations and connect case, runtime, and scheduler backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import TYPE_CHECKING

from .case import PreparedProgram, prepare_case, validate_case
from .mpi import discover_mpi_policy
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
    explicit = os.environ.get("FOAMNORDIC_MODEL_WORKER")
    candidates = [] if explicit is None else [Path(explicit)]
    prepared = os.environ.get("FOAMNORDIC_PREPARED_WORK_DIR")
    if prepared:
        candidates.append(
            Path(prepared) / "build/tools/resident/foamnordic_model_worker"
        )
    if longship is not None:
        candidates.extend(
            path / "bin/foamnordic_model_worker"
            for path in toolchain_runtime_candidates(longship.case._toolchain)
        )
    candidates.extend(
        path / "bin/foamnordic_model_worker"
        for path in active_runtime_candidates()
    )
    candidates.append(
        Path.home() / ".local/share/foamnordic/native/bin/foamnordic_model_worker"
    )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "ModelHost executable is unavailable. Set FOAMNORDIC_MODEL_WORKER "
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
    solver_command = quote_command(solver)
    if local_mpi and longship.case.ranks > 1:
        runtime_dir = next(
            iter(toolchain_runtime_candidates(longship.case._toolchain)),
            None,
        )
        policy = discover_mpi_policy(
            wrapper=longship.case._toolchain.wrapper,
            runtime_dir=runtime_dir,
        )
        if policy.launcher is not None and policy.isolate:
            environment += (
                'export FOAMNORDIC_SOLVER_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"; '
                "unset DYLD_LIBRARY_PATH OPAL_PREFIX MPI_ARCH_PATH; "
            )
            solver_command = (
                f"{shlex.quote(str(policy.launcher))} "
                '-x DYLD_LIBRARY_PATH="$FOAMNORDIC_SOLVER_DYLD_LIBRARY_PATH" '
                f"-np {longship.case.ranks} {solver_command}"
            )
        elif policy.launcher is not None:
            solver_command = quote_command(
                [policy.launcher, "-np", str(longship.case.ranks), *solver]
            )
        else:
            solver_command = quote_command(
                ["mpirun", "-np", str(longship.case.ranks), *solver]
            )
    return toolchain_shell(
        longship.case._toolchain,
        environment + "exec " + solver_command,
    )


def _artifact_metadata(path: Path) -> dict[str, object]:
    if _native is None:
        raise RuntimeError("model launch requires a FoamNordic binary wheel")
    return dict(_native.read_model_manifest(str(path.expanduser().resolve())))


def _host_command(
    longship: Longship,
    prepared: PreparedProgram,
    model_threads: int = 1,
    connections: int | None = None,
) -> tuple[str, ...]:
    if model_threads < 1:
        raise ValueError("model_threads must be positive")
    if connections is not None and connections < 1:
        raise ValueError("connections must be positive")
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
    elif model_format in {"compiled", "joblib", "equinox"}:
        executable = [sys.executable, "-m", "foamnordic.execution.resident"]
    else:
        raise ValueError(f"unsupported field-program artifact format: {model_format}")
    values: list[object] = [
        *executable,
        f"unix://{prepared.socket}",
        artifact,
        "--connections",
        str(connections or longship.case.ranks),
        "--threads",
        str(model_threads),
        "--ready-file",
        prepared.ready,
    ]
    if model_format in {"joblib", "equinox"}:
        values.extend(("--key", program.key.to_json(), "--program", program.name))
    if longship.placement.data_path == "uds":
        values.append("--no-shm")
    # ModelHost is a FoamNordic backend process, not an OpenFOAM
    # application.  In particular, wrapping it with OpenFOAM.app on macOS
    # leaves an unnecessary outer process alive after the worker exits.  The
    # solver alone owns the case toolchain; backend executables carry their
    # own runtime paths or use the active Python environment.
    return tuple(str(value) for value in values)


def _host_group_command(
    longship: Longship,
    prepared: tuple[PreparedProgram, ...],
    work_dir: Path,
    host_cpus: int,
    connections: int | None = None,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    if host_cpus < 1:
        raise ValueError("host_cpus must be positive")
    if host_cpus < len(prepared):
        raise ValueError(
            "model cpus_per_task must be at least the number of field programs"
        )
    quotient, remainder = divmod(host_cpus, max(1, len(prepared)))
    thread_budgets = tuple(
        quotient + (index < remainder) for index in range(len(prepared))
    )
    commands = [
        (
            _host_command(longship, item, threads)
            if connections is None
            else _host_command(
                longship,
                item,
                threads,
                connections=connections,
            )
        )
        for item, threads in zip(prepared, thread_budgets, strict=True)
    ]
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


def _node_ready_path(path: Path, node_index: int) -> Path:
    """Return one shared-filesystem readiness marker per attached node."""

    return path.with_name(f"{path.name}.node{node_index}")


def _multi_node_host_command(
    host: tuple[str, ...],
    ready_files: tuple[Path, ...],
    work_dir: Path,
    nodes: int,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    """Wrap an attached host command with Slurm node-local specialization."""

    if nodes < 2:
        return host, ready_files
    configuration = _internal_path(work_dir, "node-host.json")
    configuration.write_text(
        json.dumps(
            {
                "command": list(host),
                "nodes": nodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    expanded_ready = tuple(
        _node_ready_path(path, node_index)
        for path in ready_files
        for node_index in range(nodes)
    )
    return (
        sys.executable,
        "-m",
        "foamnordic.execution.node_host",
        str(configuration),
    ), expanded_ready


def launch(
    longship: Longship,
    *,
    readiness_timeout: float = 120.0,
    termination_grace: float = 30.0,
    orphan_timeout: float = 30.0,
    verbose: bool = True,
) -> Run:
    """Compile, prepare, and start one non-blocking coupled workload."""

    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    if orphan_timeout < 0:
        raise ValueError("orphan_timeout must not be negative")
    validate_case(longship)
    plan = longship.compile()
    runtime = plan.as_dict()["runtime"]
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
        _host_group_command(
            longship,
            prepared,
            work_dir,
            int(runtime["host_cpus_per_task"]),
            int(runtime["solver_tasks_per_node"]),
        )
        if prepared
        else (None, ())
    )
    if host is not None and longship.scheduler is not None:
        host, ready_files = _multi_node_host_command(
            host,
            ready_files,
            work_dir,
            longship.scheduler.nodes,
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
    batch = write_batch(
        longship,
        runtime,
        work_dir,
        host,
        solver,
        ready_files,
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
        orphan_timeout=orphan_timeout,
    )
