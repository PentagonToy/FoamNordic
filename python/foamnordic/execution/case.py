"""Isolated OpenFOAM case validation and dictionary rendering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from typing import TYPE_CHECKING

from ..contracts import adapter_contract
from ..core.expressions import FieldExpression
from ..core.layout import FieldLayout, field_layout
from ..core.managed import mark_generated
from ..core.plan import CompiledPlan
from .run import (
    _initialize_sailing_log,
    _internal_path,
    _sailing_paths,
)
from .shell import quote_command, toolchain_shell

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def validate_case(longship: Longship) -> None:
    closures = longship.closure_programs
    programs = longship.field_programs
    if len(closures) > 1:
        raise NotImplementedError(
            "one solver closure may be active at a time; use multiple Transform "
            "programs for independent field exchanges"
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
    if len(longship.observations) > 1:
        raise NotImplementedError("launch currently supports one observation schedule")
    _validate_observation_fields(longship)
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
        and closures
        and closures[0].name == "kEqnFjord"
        and not (initial / "k").is_file()
    ):
        raise FileNotFoundError("kEqnFjord requires a source-case 0/k field")
    if longship.placement.data_path not in {"auto", "shm", "uds"}:
        raise NotImplementedError(
            "attached launch currently supports auto, shm, and uds; central UCX "
            "remains an explicit HPC validation topology"
        )


def _validate_observation_fields(longship: Longship) -> None:
    """Reject unavailable observation fields before launching the workload."""

    if not longship.observations:
        return

    supported_classes = {
        "volScalarField",
        "volVectorField",
        "volSphericalTensorField",
        "volSymmTensorField",
        "volTensorField",
    }
    available = {
        name
        for name, metadata in longship.case.fields.items()
        if metadata.field_class in supported_classes
    }
    for closure in longship.closure_programs:
        contract = adapter_contract(closure.name)
        if contract is not None:
            available.update(contract.outputs)

    requested = set(longship.observations[0].summaries)
    missing = sorted(requested - available)
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(sorted(available)) or "none"
        raise ValueError(
            f"Observe requested unavailable OpenFOAM field(s): {missing_text}. "
            f"Available observable fields: {available_text}"
        )


def _default_template() -> str:
    packaged = files("foamnordic").joinpath(
        "templates/openfoam/turbulenceProperties.fjord.in"
    )
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        _REPOSITORY_ROOT
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
        _REPOSITORY_ROOT
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
        _REPOSITORY_ROOT
        / "src/foamnordic/template/openfoam/fjordExchange.in"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError("FoamNordic transform template is unavailable")


def _decomposition_template() -> str:
    packaged = files("foamnordic").joinpath(
        "templates/openfoam/decomposeParDict.in"
    )
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        _REPOSITORY_ROOT
        / "src/foamnordic/template/openfoam/decomposeParDict.in"
    )
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise RuntimeError("FoamNordic decomposition template is unavailable")


def _decomposition_subdomains(path: Path) -> int | None:
    """Read a literal numberOfSubdomains from a conventional OpenFOAM dict."""

    if not path.is_file():
        return None
    text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"//.*$", "", text, flags=re.M)
    match = re.search(r"\bnumberOfSubdomains\s+([0-9]+)\s*;", text)
    return int(match.group(1)) if match is not None else None


def _prepare_decomposition(path: Path, ranks: int) -> bool:
    """Keep a compatible user policy or install a portable run-local policy.

    Method-specific coefficients (for example hierarchical ``n``) are coupled
    to ``numberOfSubdomains``.  Mutating only the latter produces an invalid
    dictionary, so a rank mismatch is normalized to ``scotch`` in the copied
    run case.  The source case is never modified.
    """

    if _decomposition_subdomains(path) == ranks:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _decomposition_template()
        .replace("@NUMBER_OF_SUBDOMAINS@", str(ranks))
        .replace("@DECOMPOSITION_METHOD@", "scotch")
        .replace("@METHOD_COEFFICIENTS@", ""),
        encoding="utf-8",
    )
    return True


def _derived_scheme_defaults() -> dict[str, dict[str, str]]:
    packaged = files("foamnordic").joinpath(
        "templates/openfoam/derivedSchemes.json"
    )
    source = (
        _REPOSITORY_ROOT
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
    for program in longship.field_programs:
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
        "FOAMNORDIC_OBSERVATION_EVERY": observation.interval,
        "FOAMNORDIC_OBSERVATION_OFFSET": 0,
        "FOAMNORDIC_OBSERVATION_MAX_RECORDS": 64,
        "FOAMNORDIC_OBSERVATION_MAX_BYTES": 256 * 1024,
        "FOAMNORDIC_OBSERVATION_OVERFLOW": "dropOldest",
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


def _closure_values(
    closure: Closure,
    address: str,
    shared: bool,
    observation: str,
) -> dict[str, str]:
    return {
        "FOAMNORDIC_ADDRESS": address,
        "FOAMNORDIC_SHARED_MEMORY": str(shared).lower(),
        "FOAMNORDIC_INPUT_KEYS": "\n                ".join(closure.inputs),
        "FOAMNORDIC_INPUT_EXPRESSIONS": "\n                ".join(
            _expression(value) for value in closure.inputs.values()
        ),
        "FOAMNORDIC_OUTPUT_FIELDS": "\n                ".join(
            str(value.field_name) for value in closure.outputs.values()
        ),
        "FOAMNORDIC_OUTPUT_KEYS": "\n                ".join(closure.outputs),
        "FOAMNORDIC_OBSERVATION_BLOCK": observation,
    }


def _closure_body(
    closure: Closure,
    address: str,
    shared: bool,
    observation: str = "",
) -> str:
    values = _closure_values(closure, address, shared, observation)
    return f'''address          "{values["FOAMNORDIC_ADDRESS"]}";
        sessionId        1;
        sharedMemory     {values["FOAMNORDIC_SHARED_MEMORY"]};
        ucx              false;

        inputs
        (
                {values["FOAMNORDIC_INPUT_KEYS"]}
        );

        inputExpressions
        (
                {values["FOAMNORDIC_INPUT_EXPRESSIONS"]}
        );

        outputs
        (
                {values["FOAMNORDIC_OUTPUT_FIELDS"]}
        );

        outputKeys
        (
                {values["FOAMNORDIC_OUTPUT_KEYS"]}
        );

        {values["FOAMNORDIC_OBSERVATION_BLOCK"]}'''.strip()


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
    mapped_outputs = any(
        logical_name != expression.field_name
        for logical_name, expression in closure.outputs.items()
    )
    if mapped_outputs and "@FOAMNORDIC_OUTPUT_KEYS@" not in template:
        raise ValueError(
            "an integration template with logical output names must place "
            "@FOAMNORDIC_OUTPUT_KEYS@ beside @FOAMNORDIC_OUTPUT_FIELDS@"
        )
    variables = {
        "FOAMNORDIC_MODEL": closure.name,
        "FOAMNORDIC_MODEL_CONFIGURATION": model_configuration,
        **_closure_values(
            closure,
            address,
            shared,
            ""
            if observation_path is None
            else _observation_block(longship, observation_path),
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
    observation_block: str = "",
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
        "FOAMNORDIC_OBSERVATION_BLOCK": observation_block,
    }
    rendered = _transform_template()
    for name, value in values.items():
        rendered = rendered.replace(f"@{name}@", str(value))
    unresolved = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved OpenFOAM transform variables: {unresolved}")
    return rendered.strip()


_FIELD_LAYOUTS = {
    "volScalarField": "scalar",
    "volVectorField": "vector",
    "volSphericalTensorField": "spherical_tensor",
    "volSymmTensorField": "symm_tensor",
    "volTensorField": "tensor",
}


def _field_layout(longship: Longship, expression: FieldExpression) -> FieldLayout:
    """Resolve a stored field layout from the source-case header."""

    assert expression.field_name is not None
    name = expression.field_name
    if name in {"x", "y", "z"}:
        return field_layout("scalar")
    base, separator, component = name.partition(".")
    metadata = longship.case.field(base)
    try:
        layout = field_layout(_FIELD_LAYOUTS[metadata.field_class])
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
        return field_layout("scalar")
    return layout


def _expression_layout(
    longship: Longship, expression: FieldExpression
) -> FieldLayout:
    """Infer the physical and packed layout of an OpenFOAM expression."""

    if expression.operation == "field":
        return _field_layout(longship, expression)
    if expression.operation == "filter_width":
        return field_layout("scalar")

    if expression.field_name is not None:
        arguments = (FieldExpression("field", expression.field_name),)
    else:
        arguments = expression.arguments
    layouts = tuple(_expression_layout(longship, value) for value in arguments)
    kinds = tuple(value.kind for value in layouts)
    operation = expression.operation

    if operation == "grad":
        if kinds == ("scalar",):
            return field_layout("vector")
        if kinds == ("vector",):
            return field_layout("tensor")
        raise ValueError("grad() function inputs support scalar or vector fields")
    if operation == "div":
        if len(layouts) == 2:
            return layouts[1]
        if kinds == ("vector",):
            return field_layout("scalar")
        if kinds in {("symm_tensor",), ("tensor",)}:
            return field_layout("vector")
        raise ValueError("div() function input has an unsupported field shape")
    if operation == "laplacian":
        return layouts[-1]
    if operation == "curl":
        if kinds != ("vector",):
            raise ValueError("curl() function input requires a vector field")
        return field_layout("vector")
    if operation in {"mag", "ddot"}:
        return field_layout("scalar")
    if operation == "symm":
        if kinds != ("tensor",):
            raise ValueError("symm() function input requires a tensor field")
        return field_layout("symm_tensor")
    if operation == "dev":
        if kinds not in {("scalar",), ("symm_tensor",), ("tensor",)}:
            raise ValueError("dev() function input requires a scalar or tensor field")
        return layouts[0]
    if operation == "dot":
        dot_kinds = {
            ("vector", "vector"): "scalar",
            ("tensor", "vector"): "vector",
            ("symm_tensor", "vector"): "vector",
            ("vector", "tensor"): "vector",
            ("vector", "symm_tensor"): "vector",
            ("tensor", "tensor"): "tensor",
            ("tensor", "symm_tensor"): "tensor",
            ("symm_tensor", "tensor"): "tensor",
            ("symm_tensor", "symm_tensor"): "tensor",
        }
        try:
            return field_layout(dot_kinds[kinds])
        except KeyError:
            raise ValueError(
                "dot() function inputs have unsupported field shapes"
            ) from None
    raise ValueError(
        f"cannot infer function input width for OpenFOAM operation {operation!r}"
    )


def _output_layout(
    longship: Longship,
    program: Closure | Transform,
    expression: FieldExpression,
) -> FieldLayout:
    """Resolve a mutable output from its case or built-in adapter contract."""

    try:
        return _expression_layout(longship, expression)
    except KeyError:
        if expression.operation != "field" or expression.field_name is None:
            raise
        contract = adapter_contract(program.name)
        if contract is not None:
            try:
                return contract.outputs[expression.field_name]
            except KeyError:
                pass
        raise KeyError(
            f"cannot infer output field {expression.field_name!r} for "
            f"case-sensitive adapter {program.name!r}; add the field to "
            "the source case, declare a built-in adapter contract, or use "
            "a model artifact with an explicit contract"
        ) from None


def _output_width(
    longship: Longship,
    program: Closure | Transform,
    expression: FieldExpression,
) -> int:
    """Return the packed width of a mutable output field."""

    return _output_layout(longship, program, expression).transport_width


def _package_function(
    longship: Longship,
    program: Closure | Transform,
    work_dir: Path,
) -> Path | None:
    """Serialize one direct function and its inferred native tensor contract."""

    if program.operator.kind != "function":
        return None
    try:
        import cloudpickle
        from .. import _native
    except ImportError as error:
        raise RuntimeError(
            "Operator.function requires cloudpickle and a FoamNordic binary wheel"
        ) from error
    internal = _internal_path(work_dir, "function")
    internal.mkdir(parents=True, exist_ok=True)
    payload = internal / f"{program.name}.function"
    manifest = internal / f"{program.name}.fnom"
    with payload.open("wb") as stream:
        input_layouts = tuple(
            _expression_layout(longship, expression)
            for expression in program.inputs.values()
        )
        output_layouts = tuple(
            _output_layout(longship, program, expression)
            for expression in program.outputs.values()
        )
        cloudpickle.dump(
            {
                "schema": "foamnordic.function/v1",
                "function": program.operator.source,
                "inputs": tuple(program.inputs),
                "input_widths": tuple(
                    layout.transport_width for layout in input_layouts
                ),
                "input_layouts": tuple(
                    layout.to_plan() for layout in input_layouts
                ),
                "outputs": tuple(program.outputs),
                "output_layouts": tuple(
                    layout.to_plan() for layout in output_layouts
                ),
                "program": program.name,
                "key": program.key.to_plan(),
            },
            stream,
            protocol=5,
        )
    inputs = [
        (name, layout.transport_width)
        for name, layout in zip(program.inputs, input_layouts, strict=True)
    ]
    outputs = [
        (name, _output_width(longship, program, expression))
        for name, expression in program.outputs.items()
    ]
    _native.write_model_bundle(
        str(manifest),
        str(payload),
        program.name,
        "joblib",
        inputs,
        outputs,
        "float64",
        [],
        None,
        None,
    )
    payload.unlink()
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
) -> tuple[Path, Path, tuple[PreparedProgram, ...]]:
    workspace = longship.case.run_dir.expanduser().resolve()
    runs = workspace / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_name = (
        re.sub(r"[^A-Za-z0-9-]+", "-", longship.name).strip("-")
        or "FoamNordic"
    )
    for _ in range(100):
        work_dir = runs / f"{run_name}-preparing-{secrets.token_hex(4)}"
        try:
            work_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Cannot allocate a unique FoamNordic run directory")
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
    closures = longship.closure_programs
    for index, program in enumerate(closures):
        slug = _program_slug(program.name, index)
        ready = _internal_path(work_dir, f"{slug}.ready")
        socket = _socket_path(work_dir, index)
        artifact = _package_function(longship, program, work_dir)
        prepared.append(PreparedProgram(program, ready, socket, artifact))
    for offset, transform in enumerate(longship.transforms, len(prepared)):
        slug = _program_slug(transform.name, offset)
        ready = _internal_path(work_dir, f"{slug}.ready")
        socket = _socket_path(work_dir, offset)
        artifact = _package_function(longship, transform, work_dir)
        prepared.append(PreparedProgram(transform, ready, socket, artifact))
    observations = work_dir / "observations"
    if longship.observations:
        observations.mkdir(parents=True, exist_ok=True)
    if closures:
        closure_runtime = prepared[0]
        destination, contents = render_dictionary(
            longship,
            closures[0],
            f"unix://{closure_runtime.socket}",
            longship.placement.data_path != "uds",
            observations / "observations.{rank}.jsonl",
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
        observation_block = ""
        if index == observation_transform:
            observation_block = _observation_block(
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
                    observation_block,
                ),
            )
        )

    control = case_dir / "system/controlDict"
    control_path = quote_command((control,))
    # controlDict stores an OpenFOAM word, while the process command may be an
    # absolute executable selected by an advanced user.
    application = quote_command((Path(longship.case.application).name,))
    mesh_commands = _mesh_commands(longship, case_dir)
    commands = [
        *mesh_commands,
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
    if longship.field_programs:
        if openfoam_library is None:
            raise RuntimeError(
                "field-program case preparation requires the selected "
                "OpenFOAM integration library"
            )
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
    if verbose and mesh_commands:
        if longship.case._mesh == "blockMesh":
            print(
                f"[FoamNordic] Preparing mesh with blockMesh: {longship.name}"
            )
        else:
            print(
                "[FoamNordic] Validating existing mesh with checkMesh: "
                f"{longship.name}"
            )
    sailing_log, _, _ = _sailing_paths(work_dir, longship.name)
    _initialize_sailing_log(sailing_log, longship.name)
    with sailing_log.open("a", encoding="utf-8") as stream:
        stream.write("[FoamNordic] Preparing isolated OpenFOAM case\n")
    try:
        with sailing_log.open("ab") as stream:
            subprocess.run(
                toolchain_shell(longship.case._toolchain, " && ".join(commands)),
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"OpenFOAM case preparation failed; inspect {sailing_log}"
        ) from error
    if verbose and mesh_commands:
        print(f"[FoamNordic] Mesh is ready: {longship.name}")
    if longship.case.ranks > 1:
        expected = {f"processor{rank}" for rank in range(longship.case.ranks)}
        actual = {
            path.name
            for path in case_dir.iterdir()
            if path.is_dir() and re.fullmatch(r"processor\d+", path.name)
        }
        if actual != expected:
            raise RuntimeError(
                "OpenFOAM decomposition does not match Case.ranks: "
                f"expected {sorted(expected)}, found {sorted(actual)}"
            )
    return work_dir, case_dir, tuple(prepared)
