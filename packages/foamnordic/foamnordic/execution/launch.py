"""Compile declarations and connect case, runtime, and scheduler backends."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from bindnordic.models.artifact import load as load_artifact

from .case import PreparedProgram, prepare_case
from .case_validation import validate_case
from .mpi import discover_mpi_policy
from .process import launch_local, launch_process
from .run import Run
from .run_files import internal_path, sailing_paths
from .runtime_paths import active_runtime_candidates, toolchain_runtime_candidates
from .shell import quote_command, toolchain_shell
from .slurm import force_cancel, write_batch, write_submission_wrapper

try:
    from .. import _native
except ImportError:
    _native = None

if TYPE_CHECKING:
    from ..core.spec import Longship


_INHERITED_SLURM_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_")


def _programs(longship: Longship):
    return longship.field_programs


def _model_project_roots(longship: Longship) -> tuple[Path, ...]:
    """Find Python project roots that own resident model artefacts."""

    roots: list[Path] = []
    markers = ("pyproject.toml", "setup.py", "setup.cfg", ".git")
    for program in _programs(longship):
        if program.artifact is None:
            continue
        artifact = program.artifact.expanduser().resolve()
        for parent in artifact.parents:
            if any((parent / marker).exists() for marker in markers):
                if parent not in roots:
                    roots.append(parent)
                break
    return tuple(roots)


def _runtime_environment(longship: Longship | None = None) -> dict[str, str]:
    """Copy the caller environment and expose model-owned Python projects."""

    environment = dict(os.environ)
    roots = () if longship is None else _model_project_roots(longship)
    if roots:
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in (*roots, *((inherited,) if inherited else ()))
        )
    return environment


def _submission_environment(longship: Longship | None = None) -> dict[str, str]:
    """Copy the caller environment without a parent Slurm allocation."""

    return {
        key: value
        for key, value in _runtime_environment(longship).items()
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
    return load_artifact(path).as_dict()


def _host_command(
    longship: Longship,
    prepared: PreparedProgram,
    model_threads: int = 1,
    connections: int | None = None,
    profile_every: int = 0,
    receive_memory_bytes: int | None = None,
) -> tuple[str, ...]:
    if model_threads < 1:
        raise ValueError("model_threads must be positive")
    if connections is not None and connections < 1:
        raise ValueError("connections must be positive")
    if profile_every < 0:
        raise ValueError("profile_every must not be negative")
    if receive_memory_bytes is not None and receive_memory_bytes < 1:
        raise ValueError("receive_memory_bytes must be positive")
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
    elif model_format in {"compiled", "joblib", "equinox", "table"}:
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
        "--profile-every",
        str(profile_every),
        "--program",
        program.name,
        "--ready-file",
        prepared.ready,
    ]
    if model_format in {"joblib", "equinox"}:
        values.extend(("--key", program.key.to_json()))
    if receive_memory_bytes is not None:
        values.extend(("--receive-bytes", str(receive_memory_bytes)))
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
    profile_every: int = 0,
    receive_memory_bytes: int | None = None,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    if host_cpus < 1:
        raise ValueError("host_cpus must be positive")
    thread_budgets = (host_cpus,) * len(prepared)
    commands = [
        (
            _host_command(
                longship,
                item,
                threads,
                profile_every=profile_every,
                receive_memory_bytes=receive_memory_bytes,
            )
            if connections is None
            else _host_command(
                longship,
                item,
                threads,
                connections=connections,
                profile_every=profile_every,
                receive_memory_bytes=receive_memory_bytes,
            )
        )
        for item, threads in zip(prepared, thread_budgets, strict=True)
    ]
    if len(commands) == 1:
        return commands[0], (prepared[0].ready,)
    programs = []
    plugins: list[str] = []
    for command in commands:
        if len(command) >= 4 and command[:3] == (
            sys.executable,
            "-m",
            "foamnordic.execution.resident",
        ):
            programs.append(list(command[3:]))
        else:
            programs.append(list(command[1:]))
            if len(command) >= 3:
                metadata = _artifact_metadata(Path(command[2]))
                if str(metadata["format"]) == "onnx":
                    prefix = Path(command[0]).resolve().parent.parent
                    candidates = (
                        prefix / "lib/libfoamnordic_onnx_connector.dylib",
                        prefix / "lib/libfoamnordic_onnx_connector.so",
                        prefix / "lib64/libfoamnordic_onnx_connector.so",
                    )
                    plugin = next((path for path in candidates if path.is_file()), None)
                    if plugin is None:
                        raise RuntimeError(
                            "The ONNX connector plugin is unavailable; rebuild the "
                            "active FoamNordic runtime."
                        )
                    if str(plugin) not in plugins:
                        plugins.append(str(plugin))
    aggregate = internal_path(work_dir, "programs.ready")
    configuration = internal_path(work_dir, "host-programs.json")
    configuration.write_text(
        json.dumps(
            {
                "programs": programs,
                "capacity_units": host_cpus,
                "ready_files": [str(item.ready) for item in prepared],
                "aggregate_ready": str(aggregate),
                "connector_plugins": plugins,
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
    configuration = internal_path(work_dir, "node-host.json")
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
    profile_every: int = 0,
    receive_memory_bytes: int | None = None,
) -> Run:
    """Compile, prepare, and start one non-blocking coupled workload."""

    if readiness_timeout <= 0 or termination_grace <= 0:
        raise ValueError("lifecycle timeouts must be positive")
    if orphan_timeout < 0:
        raise ValueError("orphan_timeout must not be negative")
    if profile_every < 0:
        raise ValueError("profile_every must not be negative")
    if receive_memory_bytes is not None and receive_memory_bytes < 1:
        raise ValueError("receive_memory_bytes must be positive")
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
        profile_every=profile_every,
    )
    resident_programs = tuple(
        item for item in prepared if item.native_program is None
    )
    def pin_runtime() -> tuple[Path, ...]:
        if integration_library is None:
            return ()
        generation = integration_library.resolve().parent.parent
        if generation.parent.name != "generations":
            return ()
        from ..build.generations import pin

        return (pin(generation, str(work_dir)),)
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
            resident_programs,
            work_dir,
            int(runtime["host_cpus_per_task"]),
            int(runtime["solver_tasks_per_node"]),
            profile_every,
            receive_memory_bytes,
        )
        if resident_programs
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
        longship_log, host_log, solver_log = sailing_paths(work_dir, longship.name)
        host_log.touch()
        return launch_process(
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
        pin_cleanup = pin_runtime()
        return launch_local(
            host=host,
            solver=solver,
            ready_files=ready_files,
            work_dir=work_dir,
            readiness_timeout=readiness_timeout,
            termination_grace=termination_grace,
            plan_digest=plan.digest,
            name=longship.name,
            observation_sources=longship.case.ranks,
            cleanup_paths=pin_cleanup,
            environment=_runtime_environment(longship),
        )

    for command in ("sbatch", "squeue", "sacct", "scancel"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Slurm command is unavailable: {command}")
    pin_cleanup = pin_runtime()
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
    longship_log, host_log, solver_log = sailing_paths(work_dir, longship.name)
    assert longship.scheduler is not None
    return launch_process(
        (wrapper,),
        work_dir=work_dir,
        process_log=internal_path(work_dir, "submission.log"),
        longship_log=longship_log,
        host_log=host_log,
        solver_log=solver_log,
        plan_digest=plan.digest,
        name=longship.name,
        job_file=internal_path(work_dir, "job.id"),
        partition=longship.scheduler.partition or "auto",
        force_cancel=lambda: force_cancel(work_dir),
        observation_sources=longship.case.ranks,
        environment=_submission_environment(longship),
        orphan_timeout=orphan_timeout,
        cleanup_paths=pin_cleanup,
    )
