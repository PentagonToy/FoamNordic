"""Infer FoamNordic layouts from OpenFOAM field expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import adapter_contract
from ..core.expressions import FieldExpression
from ..core.layout import FieldLayout, field_layout

if TYPE_CHECKING:
    from ..core.spec import Closure, Longship, Transform


_FIELD_LAYOUTS = {
    "volScalarField": "scalar",
    "volVectorField": "vector",
    "volSphericalTensorField": "spherical_tensor",
    "volSymmTensorField": "symm_tensor",
    "volTensorField": "tensor",
    "surfaceScalarField": "scalar",
    "surfaceVectorField": "vector",
    "surfaceSphericalTensorField": "spherical_tensor",
    "surfaceSymmTensorField": "symm_tensor",
    "surfaceTensorField": "tensor",
}


def field_layout_for(
    longship: Longship,
    expression: FieldExpression,
) -> FieldLayout:
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
            "x",
            "y",
            "z",
        }:
            raise ValueError(
                f"component selector {name!r} requires U.x/U.y/U.z-style "
                "access to a volVectorField"
            )
        return field_layout("scalar")
    return layout


def expression_layout(
    longship: Longship,
    expression: FieldExpression,
) -> FieldLayout:
    """Infer the physical and packed layout of an OpenFOAM expression."""

    if expression.operation == "field":
        return field_layout_for(longship, expression)
    if expression.operation in {"filter_width", "delta_time", "constant"}:
        return field_layout("scalar")

    if expression.field_name is not None:
        arguments = (FieldExpression("field", expression.field_name),)
    else:
        arguments = expression.arguments
    layouts = tuple(expression_layout(longship, value) for value in arguments)
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
        if len(layouts) == 2 and kinds[0] != "scalar":
            raise ValueError(
                "laplacian() diffusivity expression requires a scalar field"
            )
        return layouts[-1]
    if operation == "curl":
        if kinds != ("vector",):
            raise ValueError("curl() function input requires a vector field")
        return field_layout("vector")
    if operation in {"mag", "ddot"}:
        return field_layout("scalar")
    if operation == "trace":
        if kinds not in {("tensor",), ("symm_tensor",), ("spherical_tensor",)}:
            raise ValueError("trace() function input requires a tensor field")
        return field_layout("scalar")
    if operation in {"abs", "sqrt"}:
        if kinds != ("scalar",):
            raise ValueError(f"{operation}() function input requires a scalar field")
        return field_layout("scalar")
    if operation == "greater":
        if kinds != ("scalar", "scalar"):
            raise ValueError("greater-than function inputs require scalar fields")
        return field_layout("scalar")
    if operation == "where":
        if kinds[0] != "scalar":
            raise ValueError("where() condition requires a scalar field")
        branch_constants = tuple(
            value.operation == "constant" for value in arguments[1:]
        )
        if branch_constants == (True, True):
            return field_layout("scalar")
        if branch_constants[0]:
            return layouts[2]
        if branch_constants[1]:
            return layouts[1]
        if kinds[1] != kinds[2]:
            raise ValueError("where() branches require matching field shapes")
        return layouts[1]
    if operation in {"add", "subtract", "maximum", "minimum"}:
        if any(value.operation == "constant" for value in arguments):
            if operation in {"maximum", "minimum"} and kinds == (
                "scalar",
                "scalar",
            ):
                return field_layout("scalar")
            raise ValueError(
                f"{operation}() does not accept dimensionless scalar constants"
            )
        if operation in {"maximum", "minimum"} and kinds != (
            "scalar",
            "scalar",
        ):
            raise ValueError(f"{operation}() function inputs require scalar fields")
        if kinds[0] != kinds[1]:
            raise ValueError(
                f"{operation}() function inputs require matching field shapes"
            )
        return layouts[0]
    if operation == "transport":
        raise ValueError(
            "Field.transport is a solver control node, not a packed field input"
        )
    if operation in {"multiply", "divide"}:
        constant = tuple(value.operation == "constant" for value in arguments)
        if operation == "divide" and constant == (True, False):
            if kinds[1] != "scalar":
                raise ValueError(
                    "divide() constant numerator requires a scalar field"
                )
            return field_layout("scalar")
        if constant == (True, False):
            return layouts[1]
        if constant == (False, True):
            return layouts[0]
        if operation == "multiply":
            if kinds[0] == "scalar":
                return layouts[1]
            if kinds[1] == "scalar":
                return layouts[0]
        elif kinds[1] == "scalar":
            return layouts[0]
        raise ValueError(f"{operation}() requires at least one scalar operand")
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
    if operation == "outer":
        if kinds != ("vector", "vector"):
            raise ValueError("outer() function inputs require vector fields")
        return field_layout("tensor")
    raise ValueError(
        f"cannot infer function input width for OpenFOAM operation {operation!r}"
    )


def output_layout(
    longship: Longship,
    program: Closure | Transform,
    expression: FieldExpression,
) -> FieldLayout:
    """Resolve a mutable output from its case or built-in adapter contract."""

    try:
        return expression_layout(longship, expression)
    except KeyError:
        if expression.operation != "field" or expression.field_name is None:
            raise
        contract = adapter_contract(program.name)
        if contract is not None:
            try:
                return contract.outputs[expression.field_name]
            except KeyError:
                pass
        if program.name == "kEqnFjord" and expression.field_name in {
            "kProduction",
            "kDissipationCoeff",
        }:
            return field_layout("scalar")
        raise KeyError(
            f"cannot infer output field {expression.field_name!r} for "
            f"case-sensitive adapter {program.name!r}; add the field to "
            "the source case, declare a built-in adapter contract, or use "
            "a model artifact with an explicit contract"
        ) from None


def output_width(
    longship: Longship,
    program: Closure | Transform,
    expression: FieldExpression,
) -> int:
    """Return the packed width of a mutable output field."""

    return output_layout(longship, program, expression).transport_width


__all__ = ["expression_layout", "field_layout_for", "output_layout", "output_width"]
