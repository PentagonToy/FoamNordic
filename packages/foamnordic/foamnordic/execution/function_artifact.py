"""Package direct Python functions for resident execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bindnordic.models.artifact import _write_bundle

from .openfoam_layouts import expression_layout, output_layout, output_width
from .run_files import internal_path

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


def package_function(
    longship: Longship,
    program: Closure | Transform,
    work_dir: Path,
) -> Path | None:
    """Serialize one direct function and its inferred native tensor contract."""

    if program.operator.kind != "function":
        return None
    try:
        import cloudpickle

        from .. import _native  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Operator.function requires cloudpickle and a FoamNordic binary wheel"
        ) from error
    internal = internal_path(work_dir, "function")
    internal.mkdir(parents=True, exist_ok=True)
    payload = internal / f"{program.name}.function"
    manifest = internal / f"{program.name}.fnom"
    input_layouts = tuple(
        expression_layout(longship, expression)
        for expression in program.inputs.values()
    )
    output_layouts = tuple(
        output_layout(longship, program, expression)
        for expression in program.outputs.values()
    )
    with payload.open("wb") as stream:
        cloudpickle.dump(
            {
                "schema": "foamnordic.function",
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
        (name, output_width(longship, program, expression))
        for name, expression in program.outputs.items()
    ]
    _write_bundle(
        str(manifest),
        str(payload),
        program.name,
        "joblib",
        inputs,
        outputs,
        "float64",
        [],
    )
    payload.unlink()
    return manifest


__all__ = ["package_function"]
