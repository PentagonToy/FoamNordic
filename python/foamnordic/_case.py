"""Isolated OpenFOAM case validation and dictionary rendering."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

from ._managed import mark_generated
from ._plan import CompiledPlan
from ._run import _internal_path
from ._shell import quote_command, toolchain_shell

if TYPE_CHECKING:
    from ._expressions import FieldExpression
    from ._spec import Closure, Longship


def validate_case(longship: Longship) -> None:
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
    if len(longship.closures) > 1:
        raise NotImplementedError("launch currently supports at most one ClosureHost artifact")
    if len(longship.observations) > 1:
        raise NotImplementedError("launch currently supports one observation schedule")
    if not longship.closures:
        if longship.observations:
            raise NotImplementedError(
                "pure OpenFOAM observation requires the forthcoming function-object hook"
            )
        return
    closure = longship.closures[0]
    if not closure.artifact.expanduser().is_file():
        raise FileNotFoundError(f"closure artifact does not exist: {closure.artifact}")
    if (
        longship.case.integration is None
        and closure.name == "kEqnFjord"
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
    for closure in longship.closures:
        for expression in closure.inputs.values():
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


def prepare_case(
    longship: Longship,
    plan: CompiledPlan,
) -> tuple[Path, Path, Path]:
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
    ready = _internal_path(work_dir, "closure.ready")
    if longship.closures:
        observations = work_dir / "observations"
        if longship.observations:
            observations.mkdir(parents=True, exist_ok=True)
        address = f"unix://{_internal_path(work_dir, 'closure.sock')}"
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

    control = case_dir / "system/controlDict"
    control_path = quote_command((control,))
    application = quote_command((longship.case.application,))
    commands = [
        f"foamDictionary {control_path} -entry application -set {application}",
    ]
    commands.extend(_scheme_commands(longship, case_dir))
    if longship.closures:
        libraries = quote_command(('("libfoamnordicOpenFOAM.so")',))
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
    return work_dir, case_dir, ready
