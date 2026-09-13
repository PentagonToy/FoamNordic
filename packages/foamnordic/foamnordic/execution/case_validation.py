"""Validate an OpenFOAM case against its FoamNordic programs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import adapter_contract
from ..core.expressions import FieldExpression
from ..core.layout import FieldLayout
from ..models import load as load_model
from .openfoam_layouts import expression_layout, output_layout

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


def validate_case(longship: Longship) -> None:
    closures = longship.closure_programs
    programs = longship.field_programs
    if len(longship.closures) > 1:
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
            "solver-native FoamNordic hook; standard OpenFOAM applications expose only "
            "time_step_start and time_step_end"
        )
    source = longship.case.case_dir.resolve()
    # Resolve both paths before reporting a relative location.  On macOS,
    # /tmp and /private/tmp name the same directory but Path.relative_to()
    # quite correctly treats their lexical forms as unrelated.
    initial = longship.case.initial_directory.resolve()
    required = (
        initial,
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
        if program.artifact is not None:
            _validate_model_contract(longship, program)
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


def _validate_model_contract(longship: Longship, program: Closure | Transform) -> None:
    """Match FNOM names and packed widths to the selected OpenFOAM bindings."""

    assert program.artifact is not None
    manifest = load_model(program.artifact)
    if program.operator.kind == "table" and manifest.format != "table":
        raise ValueError(
            f"{program.name} uses Operator.table, but its FNOM format is "
            f"{manifest.format!r}"
        )
    if program.operator.kind == "model" and manifest.format == "table":
        raise ValueError(
            f"{program.name} uses Operator.model for an FNOM table; use "
            "Operator.table instead"
        )
    declared = (
        ("inputs", manifest.inputs, program.inputs),
        ("outputs", manifest.outputs, program.outputs),
    )
    for label, ports, bindings in declared:
        names = tuple(port.name for port in ports)
        if set(names) != set(bindings) or len(names) != len(bindings):
            raise ValueError(
                f"{program.name} {label} do not match its FNOM contract: "
                f"{tuple(bindings)!r} != {names!r}"
            )
        for port in ports:
            expression = bindings[port.name]
            layout = (
                _input_layout(longship, program, expression)
                if label == "inputs"
                else output_layout(longship, program, expression)
            )
            if port.components != layout.transport_width:
                raise ValueError(
                    f"{program.name} {label} port {port.name!r} has "
                    f"{port.components} FNOM component(s), but "
                    f"{expression.canonical!r} requires "
                    f"{layout.transport_width}"
                )

    adapter = adapter_contract(program.name)
    if adapter is None:
        return
    output_fields = tuple(
        str(value.field_name) for value in program.outputs.values()
    )
    if program.name == "kEqnFjord":
        compatible = {
            frozenset(("k", "nut")),
            frozenset(("nut", "kProduction", "kDissipationCoeff")),
        }
        if frozenset(output_fields) in compatible:
            return
    if set(output_fields) != set(adapter.outputs):
        raise ValueError(
            f"{program.name} requires OpenFOAM outputs {tuple(adapter.outputs)!r}; "
            f"got {output_fields!r}"
        )


def _input_layout(
    longship: Longship,
    program: Closure | Transform,
    expression: FieldExpression,
) -> FieldLayout:
    """Resolve an input layout, including adapter-owned derived fields."""

    try:
        return expression_layout(longship, expression)
    except KeyError:
        contract = adapter_contract(program.name)
        if contract is not None and expression.canonical in contract.inputs:
            return contract.inputs[expression.canonical]
        raise


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
        "surfaceScalarField",
        "surfaceVectorField",
        "surfaceSphericalTensorField",
        "surfaceSymmTensorField",
        "surfaceTensorField",
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


__all__ = ["validate_case"]
