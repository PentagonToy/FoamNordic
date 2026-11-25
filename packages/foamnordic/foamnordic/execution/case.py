"""Render and stage an isolated OpenFOAM case."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.graph import NativeAlgebraPlan, NativeFieldPlan, NativeTransportPlan
from ..core.plan import CompiledPlan
from .case_staging import prepare_staged_case, stage_case, verify_decomposition
from .function_artifact import package_function as _package_function
from .openfoam_rendering import (
    observation_block,
    render_dictionary,
    render_transform_dictionary,
    scheme_commands,
)
from .openfoam_templates import prepare_decomposition as _prepare_decomposition
from .run_files import internal_path
from .shell import quote_command

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


@dataclass(frozen=True, slots=True)
class PreparedProgram:
    """One isolated field program and its worker lifecycle endpoints."""

    program: Closure | Transform
    ready: Path
    socket: Path
    artifact: Path | None
    native_program: NativeFieldPlan | None = None


def _program_slug(name: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "program"
    return f"{index:02d}-{cleaned}"


def _socket_path(work_dir: Path, index: int) -> Path:
    """Return a node-local address safely below sockaddr_un.sun_path limits."""

    root = Path(os.environ.get("FOAMNORDIC_SOCKET_DIR", "/tmp")).expanduser()
    identity = hashlib.sha256(str(work_dir).encode("utf-8")).hexdigest()[:16]
    user = getattr(os, "getuid", lambda: 0)()
    path = root / f"fn-{user}-{identity}-{index:02d}.sock"
    if len(os.fsencode(path)) >= 104:
        raise ValueError(
            "FOAMNORDIC_SOCKET_DIR is too long for a portable Unix socket path: "
            f"{root}"
        )
    return path


_POLYMESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")


def _has_poly_mesh(case_dir: Path) -> bool:
    mesh = case_dir / "constant/polyMesh"
    return all(
        (mesh / name).is_file() or (mesh / f"{name}.gz").is_file()
        for name in _POLYMESH_FILES
    )


def _mesh_commands(longship: Longship, case_dir: Path) -> list[str]:
    mesh = longship.case._mesh
    case_path = quote_command((case_dir,))
    commands: list[str] = []
    if mesh == "blockMesh":
        dictionary = case_dir / "system/blockMeshDict"
        if not dictionary.is_file():
            raise FileNotFoundError(
                "OpenFOAM blockMesh initialization requires "
                f"system/blockMeshDict: {dictionary}"
            )
        commands.append(f"blockMesh -case {case_path}")
    elif not _has_poly_mesh(case_dir):
        raise FileNotFoundError(
            "OpenFOAM mesh is missing from constant/polyMesh. Generate it "
            "before launch or call "
            "case.initialize(mesh='blockMesh', validate_mesh=True)."
        )
    if longship.case._validate_mesh:
        commands.append(f"checkMesh -case {case_path}")
    return commands


def prepare_case(
    longship: Longship,
    plan: CompiledPlan,
    openfoam_library: Path | None = None,
    *,
    verbose: bool = False,
    profile_every: int = 0,
) -> tuple[Path, Path, tuple[PreparedProgram, ...]]:
    work_dir, case_dir = stage_case(longship, plan)
    prepared: list[PreparedProgram] = []
    closures = longship.closure_programs
    programs = (*closures, *longship.transforms)
    for index, program in enumerate(programs):
        slug = _program_slug(program.name, index)
        ready = internal_path(work_dir, f"{slug}.ready")
        socket = _socket_path(work_dir, index)
        native_program = program.native_program
        if 0 < index < len(closures) and isinstance(native_program, NativeTransportPlan):
            raise ValueError(
                "Field.transport cannot be used by a solver-mediated update stage"
            )
        artifact = (
            None
            if native_program is not None
            else _package_function(longship, program, work_dir)
        )
        prepared.append(
            PreparedProgram(program, ready, socket, artifact, native_program)
        )
        if native_program is not None and verbose:
            print(
                f"[FoamNordic] {program.name} execution lane: native OpenFOAM."
            )
    observations = work_dir / "observations"
    if longship.observations:
        observations.mkdir(parents=True, exist_ok=True)
    if closures:
        closure_runtime = prepared[0]
        update_runtimes = tuple(
            (program, f"unix://{prepared[index].socket}")
            for index, program in enumerate(closures[1:], 1)
        )
        destination, contents = render_dictionary(
            longship,
            closures[0],
            f"unix://{closure_runtime.socket}",
            longship.placement.data_path != "uds",
            observations / "observations.{rank}.jsonl",
            update_runtimes,
            profile_every,
            closure_runtime.native_program,
            momentum_transport=(case_dir / "constant/momentumTransport").is_file(),
        )
        output = case_dir / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(contents, encoding="utf-8")

    transform_dictionaries: list[tuple[Transform, str]] = []
    closure_offset = len(closures)
    observation_transform = None
    if longship.observations and not closures and longship.transforms:
        stage_order = {
            "time_step_start": 0,
            "outer_corrector": 1,
            "pressure_corrected": 2,
            "time_step_end": 3,
        }
        observation_transform = max(
            range(len(longship.transforms)),
            key=lambda item: (stage_order[longship.transforms[item].at], item),
        )
    for index, transform in enumerate(longship.transforms):
        runtime = prepared[closure_offset + index]
        observation_contents = ""
        if index == observation_transform:
            observation_contents = observation_block(
                longship,
                observations / "observations.{rank}.jsonl",
            )
        transform_dictionaries.append(
            (
                transform,
                render_transform_dictionary(
                    transform,
                    f"unix://{runtime.socket}",
                    longship.placement.data_path != "uds",
                    observation_contents,
                    profile_every,
                    runtime.native_program
                    if isinstance(runtime.native_program, NativeAlgebraPlan)
                    else None,
                ),
            )
        )

    control = case_dir / "system/controlDict"
    control_path = quote_command((control,))
    # controlDict stores an OpenFOAM word, while the process command may be an
    # absolute executable selected by an advanced user.
    application_name = Path(longship.case.application).name
    application = quote_command((application_name,))
    mesh_commands = _mesh_commands(longship, case_dir)
    commands: list[str] = []
    if programs:
        if openfoam_library is None:
            raise RuntimeError(
                "field-program case preparation requires the selected "
                "OpenFOAM integration library"
            )
        library_directory = openfoam_library.resolve().parent
        application_directory = library_directory.parent / "bin"
        commands.append(
            f"export FOAM_USER_LIBBIN={quote_command((library_directory,))} "
            f"PATH={quote_command((application_directory,))}:\"${{PATH}}\" "
            f"LD_LIBRARY_PATH={quote_command((library_directory,))}:"
            '"${LD_LIBRARY_PATH:-}"'
        )
    commands.extend(mesh_commands)
    if application_name == "foamRun":
        commands.append(
            f"if ! foamDictionary {control_path} -entry solver -value "
            ">/dev/null 2>&1; then "
            f"case $(foamDictionary {control_path} -entry application -value 2>/dev/null) in "
            "pimpleFoam) solver=incompressibleFluid ;; "
            "rhoPimpleFoam) solver=fluid ;; "
            "reactingFoam) solver=multicomponentFluid ;; "
            "*) echo 'Set solver in system/controlDict for foamRun' >&2; exit 1 ;; "
            f'esac; foamDictionary {control_path} -entry solver -add "$solver"; fi'
        )
    commands.append(
        f"foamDictionary {control_path} -entry application -set {application}"
    )
    transport = case_dir / "constant/transportProperties"
    if transport.is_file() and not (case_dir / "constant/physicalProperties").exists():
        transport_path = quote_command((transport,))
        commands.append(
            'if [ "${WM_PROJECT_VERSION:-}" = 14 ] '
            f'&& ! foamDictionary {transport_path} -entry viscosityModel -value '
            '>/dev/null 2>&1 '
            f'&& [ "$(foamDictionary {transport_path} -entry transportModel '
            '-value 2>/dev/null)" = Newtonian ]; then '
            f"foamDictionary {transport_path} -entry viscosityModel -add constant; fi"
        )
    for transform, transform_dictionary in transform_dictionaries:
        function_name = quote_command((f"functions/{transform.name}",))
        dictionary = quote_command((transform_dictionary,))
        commands.append(
            f"if ! foamDictionary {control_path} -entry functions >/dev/null 2>&1; "
            f"then foamDictionary {control_path} -entry functions -add '{{}}'; fi; "
            f"if foamDictionary {control_path} -entry {function_name} "
            f">/dev/null 2>&1; then foamDictionary {control_path} "
            f"-entry {function_name} -set {dictionary}; else foamDictionary "
            f"{control_path} -entry {function_name} -add {dictionary}; fi"
        )
    commands.extend(scheme_commands(longship, case_dir, work_dir))
    if programs:
        assert openfoam_library is not None
        library_value = str(openfoam_library.resolve()).replace('"', '\\"')
        libraries = quote_command((f'("{library_value}")',))
        commands.append(
            f"existing_libraries=$(foamDictionary {control_path} -entry libs "
            "-value 2>/dev/null || true); "
            "if [ -n \"$existing_libraries\" ]; then "
            "existing_libraries=$(printf '%s' \"$existing_libraries\" | "
            "sed 's/^[(][[:space:]]*//; s/[[:space:]]*[)]$//'); "
            f'libraries="(${{existing_libraries}} \\"{library_value}\\")"; '
            f"foamDictionary {control_path} -entry libs -set \"$libraries\"; "
            f"else foamDictionary {control_path} -entry libs -add {libraries}; fi"
        )
    if longship.case.ranks > 1:
        decomposition = case_dir / "system/decomposeParDict"
        _prepare_decomposition(decomposition, longship.case.ranks)
        decomposition_path = quote_command((decomposition,))
        commands.append(
            f"foamDictionary {decomposition_path} "
            f"-entry numberOfSubdomains -set {longship.case.ranks}"
        )
        commands.append(f"decomposePar -case {quote_command((case_dir,))} -force")
    prepare_staged_case(
        longship,
        work_dir,
        commands,
        verbose=verbose,
        mesh_commands=bool(mesh_commands),
    )
    verify_decomposition(case_dir, longship.case.ranks)
    return work_dir, case_dir, tuple(prepared)
