"""Render FoamNordic programs as OpenFOAM dictionaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.expressions import FieldExpression
from ..core.graph import NativeAlgebraPlan, NativeFieldPlan, NativeTransportPlan
from .openfoam_templates import derived_scheme_defaults as _derived_scheme_defaults
from .openfoam_templates import runtime_template
from .shell import quote_command
from .templates import read as read_template
from .templates import render as render_template

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


def _walk_expression(value: FieldExpression):
    yield value
    for argument in value.arguments:
        yield from _walk_expression(argument)


def _scheme_requirements(longship: Longship) -> tuple[tuple[str, str, str], ...]:
    """Return missing-scheme candidates without inspecting a concrete case."""

    defaults = _derived_scheme_defaults()
    requirements: dict[tuple[str, str], str] = {}
    for program in longship.field_programs:
        native = program.native_program
        expressions = list(program.inputs.values())
        if isinstance(native, NativeTransportPlan):
            requirements[("divSchemes", f"div(phi,{native.field.field_name})")] = (
                "Gauss linear"
            )
            expressions.extend(
                (
                    native.diffusivity,
                    native.source,
                    native.implicit_sink,
                    *native.outputs.values(),
                )
            )
        elif isinstance(native, NativeAlgebraPlan):
            expressions.extend(native.outputs.values())
        for expression in expressions:
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


def scheme_commands(
    longship: Longship, case_dir: Path, work_dir: Path | None = None
) -> list[str]:
    """Validate or add required schemes and record every actual mutation."""

    schemes = case_dir / "system/fvSchemes"
    path = quote_command((schemes,))
    commands = []
    changes = None if work_dir is None else quote_command(
        (work_dir / "numerics-changes.tsv",)
    )
    for section, expression, scheme in _scheme_requirements(longship):
        exact = quote_command((f"{section}/{expression}",))
        default = quote_command((f"{section}/default",))
        selected = quote_command((scheme,))
        message = quote_command(
            (f"[FoamNordic] Added ML feature scheme: {expression} -> {scheme}",)
        )
        missing = (
            f"if ! foamDictionary {path} -entry {exact} >/dev/null 2>&1; then "
            f"foamnordic_default=$(foamDictionary {path} -entry {default} "
            f"-value 2>/dev/null || true); "
            f"if [[ -z \"$foamnordic_default\" "
            f"|| \"$foamnordic_default\" == none ]]; then "
        )
        if longship.numerics == "strict":
            error = quote_command((
                f"FoamNordic strict numerics: missing {section}/{expression}",
            ))
            commands.append(missing + f"printf '%s\\n' {error} >&2; exit 2; fi; fi")
        else:
            record = ""
            if changes is not None:
                section_value = quote_command((section,))
                expression_value = quote_command((expression,))
                scheme_value = quote_command((scheme,))
                record = (
                    f"printf '%s\\t%s\\t%s\\n' {section_value} "
                    f"{expression_value} {scheme_value} >> {changes}; "
                )
            commands.append(
                missing
                + f"foamDictionary {path} -entry {exact} -set {selected}; "
                + record
                + f"printf '%s\\n' {message}; fi; fi"
            )
    return commands


def observation_block(longship: Longship, path: Path) -> str:
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
    return render_template(
        runtime_template("observation.in", kind="OpenFOAM observation"),
        values,
        kind="OpenFOAM observation",
    ).strip()


def _expression(value: object) -> str:
    expression = str(value.canonical)
    if value.derived:
        return f'"{expression}"'
    return expression


def _closure_values(
    closure: Closure,
    address: str,
    shared: bool,
    observation: str,
    profile_every: int = 0,
) -> dict[str, str]:
    return {
        "FOAMNORDIC_ADDRESS": address,
        "FOAMNORDIC_SHARED_MEMORY": str(shared).lower(),
        "FOAMNORDIC_PROFILE_EVERY": profile_every,
        "FOAMNORDIC_PROFILE_PROGRAM": closure.name,
        "FOAMNORDIC_INPUT_KEYS": "\n                ".join(closure.inputs),
        "FOAMNORDIC_INPUT_EXPRESSIONS": "\n                ".join(
            _expression(value) for value in closure.inputs.values()
        ),
        "FOAMNORDIC_INPUT_PATCHES": "\n                ".join(
            value.patch_name or "none" for value in closure.inputs.values()
        ),
        "FOAMNORDIC_OUTPUT_FIELDS": "\n                ".join(
            str(value.field_name) for value in closure.outputs.values()
        ),
        "FOAMNORDIC_OUTPUT_KEYS": "\n                ".join(closure.outputs),
        "FOAMNORDIC_OUTPUT_PATCHES": "\n                ".join(
            value.patch_name or "none" for value in closure.outputs.values()
        ),
        "FOAMNORDIC_OBSERVATION_BLOCK": observation,
    }


def _closure_body(
    closure: Closure,
    address: str,
    shared: bool,
    observation: str = "",
    profile_every: int = 0,
) -> str:
    return render_template(
        read_template(
            "openfoam/runtime/closureBody.in",
            kind="OpenFOAM closure-body",
        ),
        _closure_values(
            closure, address, shared, observation, profile_every
        ),
        kind="OpenFOAM closure body",
    ).strip()


def _closure_update_variables(
    template: str,
    updates: tuple[tuple[Closure, str], ...],
    shared: bool,
    profile_every: int = 0,
) -> dict[str, str]:
    """Render named solver-mediated stages without interpreting their physics."""

    variables: dict[str, str] = {}
    for index, (update, address) in enumerate(updates):
        body = f"FOAMNORDIC_UPDATE_{index}_BODY"
        marker = f"@{body}@"
        if marker not in template:
            raise ValueError(
                f"custom integration template must place {marker} for "
                f"closure update {update.name!r}"
            )
        variables[body] = _closure_body(
            update, address, shared, profile_every=profile_every
        )
        variables[f"FOAMNORDIC_UPDATE_{index}_NAME"] = update.name
    return variables


def render_dictionary(
    longship: Longship,
    closure: Closure,
    address: str,
    shared: bool,
    observation_path: Path | None = None,
    updates: tuple[tuple[Closure, str], ...] = (),
    profile_every: int = 0,
    native_program: NativeFieldPlan | None = None,
) -> tuple[Path, str]:
    integration = longship.case.integration
    if native_program is not None and integration is not None:
        raise ValueError(
            "solver-native field programs require the built-in integration"
        )
    if integration is None:
        if updates:
            raise ValueError(
                "closure updates require a custom OpenFOAM integration template"
            )
        template = runtime_template(
            "turbulenceProperties.fjord.in",
            kind="OpenFOAM dictionary",
        )
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
    if (
        integration is not None
        and longship.observations
        and "@FOAMNORDIC_OBSERVATION_BLOCK@" not in template
    ):
        raise ValueError(
            "custom integration template must place "
            "@FOAMNORDIC_OBSERVATION_BLOCK@ inside its closure dictionary"
        )
    mapped_outputs = any(
        logical_name != expression.field_name
        for logical_name, expression in closure.outputs.items()
    )
    if (
        integration is not None
        and mapped_outputs
        and "@FOAMNORDIC_OUTPUT_KEYS@" not in template
        and "@FOAMNORDIC_MODEL_EXECUTION@" not in template
    ):
        raise ValueError(
            "an integration template with logical output names must place "
            "@FOAMNORDIC_OUTPUT_KEYS@ beside @FOAMNORDIC_OUTPUT_FIELDS@"
        )
    selected_patches = any(
        value.patch_name is not None
        for value in (*closure.inputs.values(), *closure.outputs.values())
    )
    if integration is not None and selected_patches and (
        "@FOAMNORDIC_INPUT_PATCHES@" not in template
        or "@FOAMNORDIC_OUTPUT_PATCHES@" not in template
    ):
        raise ValueError(
            "an integration template with patch fields must place both "
            "@FOAMNORDIC_INPUT_PATCHES@ and @FOAMNORDIC_OUTPUT_PATCHES@"
        )
    update_variables = _closure_update_variables(
        template, updates, shared, profile_every
    )
    if native_program is None:
        execution_block = (
            "foamNordicClosure\n        {\n"
            + "\n".join(
                f"            {line}" if line else ""
                for line in _closure_body(
                    closure,
                    address,
                    shared,
                    ""
                    if observation_path is None
                    else observation_block(longship, observation_path),
                    profile_every,
                ).splitlines()
            )
            + "\n        }"
        )
    elif isinstance(native_program, NativeTransportPlan):
        output = native_program.outputs["nut"]
        observation = (
            ""
            if observation_path is None
            else observation_block(longship, observation_path)
        )
        transport_lines = [
            "foamNordicTransport",
            "        {",
            f'            field             {native_program.field.canonical};',
            (
                "            diffusivity       "
                f'"{native_program.diffusivity.canonical}";'
            ),
            f'            source            "{native_program.source.canonical}";',
            (
                "            implicitSink      "
                f'"{native_program.implicit_sink.canonical}";'
            ),
            f"            lowerBound        {native_program.lower_bound:.17g};",
            f'            nut                "{output.canonical}";',
            f"            profileEvery       {profile_every};",
            f"            profileProgram     {closure.name};",
        ]
        if observation:
            transport_lines.extend(
                f"            {line}" if line else ""
                for line in observation.splitlines()
            )
        transport_lines.append("        }")
        execution_block = "\n".join(transport_lines)
    else:
        assert isinstance(native_program, NativeAlgebraPlan)
        observation = (
            ""
            if observation_path is None
            else observation_block(longship, observation_path)
        )
        algebra_lines = [
            "foamNordicAlgebra",
            "        {",
            "            outputs",
            "            {",
            *(
                "                "
                f'{closure.outputs[name].field_name} "{expression.canonical}";'
                for name, expression in native_program.outputs.items()
            ),
            "            }",
            f"            profileEvery       {profile_every};",
            f"            profileProgram     {closure.name};",
        ]
        if observation:
            algebra_lines.extend(
                f"            {line}" if line else ""
                for line in observation.splitlines()
            )
        algebra_lines.append("        }")
        execution_block = "\n".join(algebra_lines)
    variables = {
        "FOAMNORDIC_MODEL": closure.name,
        "FOAMNORDIC_MODEL_CONFIGURATION": model_configuration,
        "FOAMNORDIC_MODEL_EXECUTION": execution_block,
        **_closure_values(
            closure,
            address,
            shared,
            ""
            if observation_path is None
            else observation_block(longship, observation_path),
            profile_every,
        ),
        **custom,
        **update_variables,
    }
    rendered = render_template(template, variables, kind="OpenFOAM")
    return destination, rendered


def render_transform_dictionary(
    transform: Transform,
    address: str,
    shared: bool,
    observation_block: str = "",
    profile_every: int = 0,
    native_program: NativeAlgebraPlan | None = None,
) -> str:
    """Render a generic field exchange at the supported solver boundary."""

    if native_program is not None:
        values = {
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
            "OUTPUT_FIELDS": " ".join(
                str(transform.outputs[name].field_name)
                for name in native_program.outputs
            ),
            "OUTPUT_EXPRESSIONS": " ".join(
                f'"{expression.canonical}"'
                for expression in native_program.outputs.values()
            ),
            "OBSERVATION_BLOCK": observation_block,
            "PROFILE_EVERY": profile_every,
        }
        return render_template(
            read_template(
                "openfoam/runtime/fjordAlgebra.in",
                kind="OpenFOAM native algebra transform",
            ),
            values,
            kind="OpenFOAM native algebra transform",
        ).strip()

    values = {
        "FJORD_ADDRESS": address,
        "SESSION_ID": 1,
        "SHARED_MEMORY": str(shared).lower(),
        "UCX": "false",
        "PROFILE_EVERY": profile_every,
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
        "INPUT_PATCHES": " ".join(
            value.patch_name or "none" for value in transform.inputs.values()
        ),
        "OUTPUT_KEYS": " ".join(transform.outputs),
        "OUTPUT_FIELDS": " ".join(
            str(value.field_name) for value in transform.outputs.values()
        ),
        "OUTPUT_PATCHES": " ".join(
            value.patch_name or "none" for value in transform.outputs.values()
        ),
        "FOAMNORDIC_OBSERVATION_BLOCK": observation_block,
    }
    return render_template(
        runtime_template("fjordExchange.in", kind="OpenFOAM transform"),
        values,
        kind="OpenFOAM transform",
    ).strip()


__all__ = [
    "observation_block",
    "render_dictionary",
    "render_transform_dictionary",
    "scheme_commands",
]
