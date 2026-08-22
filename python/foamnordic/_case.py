"""Isolated OpenFOAM case validation and dictionary rendering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

from .core.managed import mark_generated
from .core.plan import CompiledPlan
from .execution.run import _internal_path
from .execution.shell import quote_command, toolchain_shell

if TYPE_CHECKING:
    from .core.expressions import FieldExpression
    from .core.spec import Closure, Longship, Transform


def validate_case(longship: Longship) -> None:
    programs = (*longship.closures, *longship.transforms)
    if len(longship.closures) > 1:
        raise NotImplementedError(
            "one solver closure may be active at a time; use multiple Transform "
            "programs for independent field exchanges"
        )
    if longship.closures and longship.closures[0].operator.kind == "function":
        raise NotImplementedError(
            "Operator.function is currently a Transform operator; Closure "
            "functions require derived-expression component inference"
        )
    unsupported = next(
        (
            transform
            for transform in longship.transforms
            if transform.at in {"outer_corrector", "pressure_corrected"}
        ),
        None,
    )
    if unsupported is not None:
        raise NotImplementedError(
            f"Transform at={unsupported.at!r} requires a "
            "solver-native FoamNordic hook; stock applications expose only "
            "time_step_start and time_step_end"
        )
    source = longship.case.case_dir.resolve()
    initial = longship.case.initial_directory
    required = (
        initial / "U",
        initial / "p",
        source / "system/controlDict",
        source / "system/fvSchemes",
        source / "system/fvSolution",
        source / "constant",
    )
    missing = [str(path.relative_to(source)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"OpenFOAM source case is incomplete: {', '.join(missing)}")
    if longship.scheduler is not None and longship.scheduler.nodes != 1:
        raise NotImplementedError(
            "automatic Slurm launch currently supports one attached solver node"
        )
    if len(longship.observations) > 1:
        raise NotImplementedError("launch currently supports one observation schedule")
    if longship.transforms and longship.observations:
        raise NotImplementedError(
            "Transform observation requires the forthcoming general observation hook"
        )
    if not programs:
        if longship.observations:
            raise NotImplementedError(
                "pure OpenFOAM observation requires the forthcoming function-object hook"
            )
        return
    for program in programs:
        if program.artifact is not None and not program.artifact.expanduser().is_file():
            raise FileNotFoundError(
                f"field-program artifact does not exist: {program.artifact}"
            )
    if (
        longship.case.integration is None
        and longship.closures
        and longship.closures[0].name == "kEqnFjord"
        and not (initial / "k").is_file()
    ):
        raise FileNotFoundError("kEqnFjord requires a source-case 0/k field")
    if longship.placement.data_path not in {"auto", "shm", "uds"}:
        raise NotImplementedError(
            "attached launch currently supports auto, shm, and uds; central UCX "
            "remains an explicit HPC validation topology"
        )


def _default_template() -> str:
    packaged = files("foamnordic").joinpath(
        "templates/openfoam/turbulenceProperties.fjord.in"
    )
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).resolve().parents[2]
        / "src/foamnordic/template/openfoam/turbulenceProperties.fjord.in"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError("FoamNordic OpenFOAM dictionary template is unavailable")


def _observation_template() -> str:
    packaged = files("foamnordic").joinpath("templates/openfoam/observation.in")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).resolve().parents[2]
        / "src/foamnordic/template/openfoam/observation.in"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError("FoamNordic observation template is unavailable")


def _transform_template() -> str:
    packaged = files("foamnordic").joinpath("templates/openfoam/fjordExchange.in")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).resolve().parents[2]
        / "src/foamnordic/template/openfoam/fjordExchange.in"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError("FoamNordic transform template is unavailable")


def _derived_scheme_defaults() -> dict[str, dict[str, str]]:
    packaged = files("foamnordic").joinpath(
        "templates/openfoam/derivedSchemes.json"
    )
    source = (
        Path(__file__).resolve().parents[2]
        / "src/foamnordic/template/openfoam/derivedSchemes.json"
    )
    path = packaged if packaged.is_file() else source
    if not path.is_file():
        raise RuntimeError("FoamNordic derived-scheme template is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("derived-scheme template must contain a mapping")
    return value


def _walk_expression(value: FieldExpression):
    yield value
    for argument in value.arguments:
        yield from _walk_expression(argument)


def _scheme_requirements(longship: Longship) -> tuple[tuple[str, str, str], ...]:
    """Return missing-scheme candidates without inspecting a concrete case."""

    defaults = _derived_scheme_defaults()
    requirements: dict[tuple[str, str], str] = {}
    for program in (*longship.closures, *longship.transforms):
        for expression in program.inputs.values():
            for node in _walk_expression(expression):
                configuration = defaults.get(node.operation)
                if configuration is None:
                    continue
                section = configuration["section"]
                requirements[(section, node.canonical)] = configuration["scheme"]
    return tuple(
        (section, expression, scheme)
        for (section, expression), scheme in requirements.items()
    )


def _scheme_commands(longship: Longship, case_dir: Path) -> list[str]:
    """Add only schemes not already covered by an exact or usable default."""

    schemes = case_dir / "system/fvSchemes"
    path = quote_command((schemes,))
    commands = []
    for section, expression, scheme in _scheme_requirements(longship):
        exact = quote_command((f"{section}/{expression}",))
        default = quote_command((f"{section}/default",))
        selected = quote_command((scheme,))
        message = quote_command(
            (f"[FoamNordic] Added ML feature scheme: {expression} -> {scheme}",)
        )
        commands.append(
            f"if ! foamDictionary {path} -entry {exact} >/dev/null 2>&1; then "
            f"foamnordic_default=$(foamDictionary {path} -entry {default} "
            f"-value 2>/dev/null || true); "
            f"if [[ -z \"$foamnordic_default\" "
            f"|| \"$foamnordic_default\" == none ]]; then "
            f"foamDictionary {path} -entry {exact} -set {selected}; "
            f"printf '%s\\n' {message}; fi; fi"
        )
    return commands


def _observation_block(longship: Longship, path: Path) -> str:
    if not longship.observations:
        return ""
    observation = longship.observations[0]
    values = {
        "FOAMNORDIC_OBSERVATION_PATH": str(path),
        "FOAMNORDIC_OBSERVATION_FIELDS": " ".join(observation.summaries),
        "FOAMNORDIC_OBSERVATION_EVERY": observation.every,
        "FOAMNORDIC_OBSERVATION_OFFSET": observation.offset,
        "FOAMNORDIC_OBSERVATION_MAX_RECORDS": observation.retention.records,
        "FOAMNORDIC_OBSERVATION_MAX_BYTES": observation.retention.maximum_bytes,
        "FOAMNORDIC_OBSERVATION_OVERFLOW": (
            "dropOldest"
            if observation.retention.overflow == "drop_oldest"
            else "dropNewest"
        ),
    }
    rendered = _observation_template()
    for name, value in values.items():
        rendered = rendered.replace(f"@{name}@", str(value))
    return rendered.strip()


def _expression(value: object) -> str:
    expression = str(getattr(value, "canonical"))
    if getattr(value, "derived"):
        return f'"{expression}"'
    return expression


def render_dictionary(
    longship: Longship,
    closure: Closure,
    address: str,
    shared: bool,
    observation_path: Path | None = None,
) -> tuple[Path, str]:
    integration = longship.case.integration
    if integration is None:
        template = _default_template()
        destination = Path("constant/turbulenceProperties")
        model_configuration = (
            "Ck 0.094;\n        Ce 1.048;"
            if closure.name == "kEqnFjord"
            else ""
        )
        custom: dict[str, str] = {}
    else:
        template = integration.source.read_text(encoding="utf-8")
        destination = integration.destination
        model_configuration = ""
        custom = dict(integration.variables)
    if longship.observations and "@FOAMNORDIC_OBSERVATION_BLOCK@" not in template:
        raise ValueError(
            "custom integration template must place "
            "@FOAMNORDIC_OBSERVATION_BLOCK@ inside its closure dictionary"
        )
    variables = {
        "FOAMNORDIC_MODEL": closure.name,
        "FOAMNORDIC_MODEL_CONFIGURATION": model_configuration,
        "FOAMNORDIC_ADDRESS": address,
        "FOAMNORDIC_SHARED_MEMORY": str(shared).lower(),
        "FOAMNORDIC_INPUT_KEYS": "\n                ".join(closure.inputs),
        "FOAMNORDIC_INPUT_EXPRESSIONS": "\n                ".join(
            _expression(value) for value in closure.inputs.values()
        ),
        "FOAMNORDIC_OUTPUT_FIELDS": "\n                ".join(
            str(value.field_name) for value in closure.outputs.values()
        ),
        "FOAMNORDIC_OBSERVATION_BLOCK": (
            ""
            if observation_path is None
            else _observation_block(longship, observation_path)
        ),
        **custom,
    }
    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace(f"@{name}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved OpenFOAM template variables: {unresolved}")
    return destination, rendered


def render_transform_dictionary(
    transform: Transform,
    address: str,
    shared: bool,
) -> str:
    """Render a generic field exchange at the supported solver boundary."""

    values = {
        "FJORD_ADDRESS": address,
        "SESSION_ID": 1,
        "SHARED_MEMORY": str(shared).lower(),
        "UCX": "false",
        "EXECUTE_INTERVAL": 1,
        "EXCHANGE_STAGE": {
            "time_step_start": "timeStepStart",
            "outer_corrector": "outerCorrector",
            "pressure_corrected": "pressureCorrected",
            "time_step_end": "timeStepEnd",
        }[transform.at],
        "EXECUTE_CONTROL": (
            "timeStep" if transform.at == "time_step_start" else "none"
        ),
        "WRITE_CONTROL": (
            "timeStep" if transform.at == "time_step_end" else "none"
        ),
        "INPUT_KEYS": " ".join(transform.inputs),
        "INPUT_FIELDS": " ".join(
            str(value.field_name) for value in transform.inputs.values()
        ),
        "OUTPUT_KEYS": " ".join(transform.outputs),
        "OUTPUT_FIELDS": " ".join(
            str(value.field_name) for value in transform.outputs.values()
        ),
    }
    rendered = _transform_template()
    for name, value in values.items():
        rendered = rendered.replace(f"@{name}@", str(value))
    unresolved = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved OpenFOAM transform variables: {unresolved}")
    return rendered.strip()


_FIELD_COMPONENTS = {
    "volScalarField": 1,
    "volVectorField": 3,
    "volSphericalTensorField": 1,
    "volSymmTensorField": 6,
    "volTensorField": 9,
}


def _field_width(longship: Longship, expression: FieldExpression) -> int:
    """Resolve a stored field port width from the source-case header."""

    assert expression.field_name is not None
    name = expression.field_name
    if name in {"x", "y", "z"}:
        return 1
    base, separator, component = name.partition(".")
    metadata = longship.case.field(base)
    try:
        width = _FIELD_COMPONENTS[metadata.field_class]
    except KeyError:
        raise ValueError(
            f"unsupported OpenFOAM field class for {base!r}: "
            f"{metadata.field_class}"
        ) from None
    if separator:
        if metadata.field_class != "volVectorField" or component not in {
            "x", "y", "z"
        }:
            raise ValueError(
                f"component selector {name!r} requires U.x/U.y/U.z-style "
                "access to a volVectorField"
            )
        return 1
    return width


def _package_function(
    longship: Longship,
    transform: Transform,
    work_dir: Path,
) -> Path | None:
    """Serialize one direct function and its inferred native tensor contract."""

    if transform.operator.kind != "function":
        return None
    try:
        import cloudpickle
        from . import _native
    except ImportError as error:
        raise RuntimeError(
            "Operator.function requires cloudpickle and a FoamNordic binary wheel"
        ) from error
    internal = _internal_path(work_dir, "function")
    internal.mkdir(parents=True, exist_ok=True)
    payload = internal / f"{transform.name}.function"
    manifest = internal / f"{transform.name}.fnom"
    with payload.open("wb") as stream:
        input_widths = tuple(
            _field_width(longship, expression)
            for expression in transform.inputs.values()
        )
        cloudpickle.dump(
            {
                "schema": "foamnordic.function/v1",
                "function": transform.operator.source,
                "inputs": tuple(transform.inputs),
                "input_widths": input_widths,
                "outputs": tuple(transform.outputs),
                "program": transform.name,
                "key": transform.key.to_plan(),
            },
            stream,
            protocol=5,
        )
    inputs = list(zip(transform.inputs, input_widths, strict=True))
    outputs = [
        (name, _field_width(longship, expression))
        for name, expression in transform.outputs.items()
    ]
    _native.write_model_manifest(
        str(manifest),
        payload.name,
        transform.name,
        "joblib",
        inputs,
        outputs,
        "float64",
        [],
        None,
        None,
    )
    return manifest


@dataclass(frozen=True, slots=True)
class PreparedProgram:
    """One isolated field program and its worker lifecycle endpoints."""

    program: Closure | Transform
    ready: Path
    socket: Path
    artifact: Path | None


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


def prepare_case(
    longship: Longship,
    plan: CompiledPlan,
    openfoam_library: Path | None = None,
) -> tuple[Path, Path, tuple[PreparedProgram, ...]]:
    workspace = longship.case.run_dir.expanduser().resolve()
    runs = workspace / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{longship.name}-", dir=runs))
    mark_generated(
        work_dir,
        kind="run",
        metadata={"plan": plan.as_dict(), "plan_digest": plan.digest},
    )
    case_dir = work_dir / "case"
    shutil.copytree(longship.case.case_dir.expanduser().resolve(), case_dir)
    if not (case_dir / "0").exists() and (case_dir / "0.orig").is_dir():
        shutil.copytree(case_dir / "0.orig", case_dir / "0")
    prepared: list[PreparedProgram] = []
    for index, program in enumerate(longship.closures):
        slug = _program_slug(program.name, index)
        ready = _internal_path(work_dir, f"{slug}.ready")
        socket = _socket_path(work_dir, index)
        prepared.append(PreparedProgram(program, ready, socket, None))
    for offset, transform in enumerate(longship.transforms, len(prepared)):
        slug = _program_slug(transform.name, offset)
        ready = _internal_path(work_dir, f"{slug}.ready")
        socket = _socket_path(work_dir, offset)
        artifact = _package_function(longship, transform, work_dir)
        prepared.append(PreparedProgram(transform, ready, socket, artifact))
    if longship.closures:
        observations = work_dir / "observations"
        if longship.observations:
            observations.mkdir(parents=True, exist_ok=True)
        closure_runtime = prepared[0]
        address = f"unix://{closure_runtime.socket}"
        destination, contents = render_dictionary(
            longship,
            longship.closures[0],
            address,
            longship.placement.data_path != "uds",
            observations / "observations.{rank}.jsonl",
        )
        output = case_dir / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(contents, encoding="utf-8")

    transform_dictionaries: list[tuple[Transform, str]] = []
    closure_offset = len(longship.closures)
    for index, transform in enumerate(longship.transforms):
        runtime = prepared[closure_offset + index]
        transform_dictionaries.append(
            (
                transform,
                render_transform_dictionary(
                    transform,
                    f"unix://{runtime.socket}",
                    longship.placement.data_path != "uds",
                ),
            )
        )

    control = case_dir / "system/controlDict"
    control_path = quote_command((control,))
    application = quote_command((longship.case.application,))
    commands = [
        f"foamDictionary {control_path} -entry application -set {application}",
    ]
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
    commands.extend(_scheme_commands(longship, case_dir))
    if longship.closures or longship.transforms:
        if openfoam_library is None:
            raise RuntimeError(
                "field-program case preparation requires the selected "
                "OpenFOAM integration library"
            )
        library_value = str(openfoam_library.resolve()).replace('"', '\\"')
        libraries = quote_command((f'("{library_value}")',))
        commands.append(
            f"if foamDictionary {control_path} -entry libs >/dev/null 2>&1; "
            f"then foamDictionary {control_path} -entry libs -set {libraries}; "
            f"else foamDictionary {control_path} -entry libs -add {libraries}; fi"
        )
    if longship.case.ranks > 1:
        decomposition = case_dir / "system/decomposeParDict"
        decomposition.write_text(
            "FoamFile { format ascii; class dictionary; object decomposeParDict; }\n"
            f"numberOfSubdomains {longship.case.ranks};\nmethod scotch;\n",
            encoding="utf-8",
        )
        commands.append(f"decomposePar -case {quote_command((case_dir,))} -force")
    with _internal_path(work_dir, "prepare.log").open("wb") as stream:
        subprocess.run(
            toolchain_shell(longship.case._toolchain, " && ".join(commands)),
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return work_dir, case_dir, tuple(prepared)
