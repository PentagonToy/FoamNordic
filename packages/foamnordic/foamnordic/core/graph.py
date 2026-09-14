"""Preparation-time compilation of solver-native function graphs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

from .expressions import FieldExpression, field


def _walk(expression: FieldExpression):
    yield expression
    for argument in expression.arguments:
        yield from _walk(argument)


def _uses_patch(expressions) -> bool:
    return any(
        node.patch_name is not None
        for expression in expressions
        for node in _walk(expression)
    )


def _uses_outside(
    expression: FieldExpression,
    target: FieldExpression,
    boundary: FieldExpression,
) -> bool:
    """Return whether target occurs without crossing the supplied boundary."""

    if expression is boundary:
        return False
    if (
        expression.operation == "field"
        and target.operation == "field"
        and expression.field_name == target.field_name
        and expression.patch_name == target.patch_name
    ):
        return True
    return any(
        _uses_outside(argument, target, boundary)
        for argument in expression.arguments
    )


def _replace(
    expression: FieldExpression,
    target: FieldExpression,
    replacement: FieldExpression,
) -> FieldExpression:
    if expression == target:
        return replacement
    if not expression.arguments:
        return expression
    return FieldExpression(
        expression.operation,
        expression.field_name,
        tuple(_replace(value, target, replacement) for value in expression.arguments),
        expression.patch_name,
    )


@dataclass(frozen=True, slots=True)
class NativeTransportPlan:
    """One scalar transport barrier and its post-solve output expressions."""

    field: FieldExpression
    diffusivity: FieldExpression
    source: FieldExpression
    implicit_sink: FieldExpression
    lower_bound: float
    outputs: Mapping[str, FieldExpression]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": "transport",
            "field": self.field.to_plan(),
            "diffusivity": self.diffusivity.to_plan(),
            "source": self.source.to_plan(),
            "implicit_sink": self.implicit_sink.to_plan(),
            "lower_bound": self.lower_bound,
            "outputs": {
                name: expression.to_plan()
                for name, expression in self.outputs.items()
            },
        }


@dataclass(frozen=True, slots=True)
class NativeAlgebraPlan:
    """Output expressions evaluated entirely by the OpenFOAM adapter."""

    outputs: Mapping[str, FieldExpression]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": "algebra",
            "outputs": {
                name: expression.to_plan()
                for name, expression in self.outputs.items()
            },
        }


NativeFieldPlan: TypeAlias = NativeAlgebraPlan | NativeTransportPlan


def requires_native_transport(function: Callable[..., object]) -> bool:
    """Detect the explicit Field.transport marker without executing a function."""

    code = getattr(function, "__code__", None)
    return code is not None and "transport" in code.co_names


def compile_native_transport(
    function: Callable[..., object],
    inputs: Mapping[str, FieldExpression],
    outputs: Mapping[str, FieldExpression],
) -> NativeTransportPlan | None:
    """Trace one explicitly native function and validate its Fjord state contract."""

    if not requires_native_transport(function):
        return None
    if _uses_patch((*inputs.values(), *outputs.values())):
        raise ValueError("Field.transport does not support patch fields")
    try:
        result = function(**inputs)
    except Exception as error:
        raise ValueError(
            "Field.transport function could not be traced with symbolic inputs"
        ) from error
    if not isinstance(result, Mapping):
        raise TypeError("Field.transport function must return a mapping")
    traced = dict(result)
    if set(traced) != {"k", "nut"}:
        raise ValueError(
            "kEqnFjord native function must return exactly 'k' and 'nut'"
        )
    if set(outputs) != {"k", "nut"}:
        raise ValueError("kEqnFjord outputs must be exactly 'k' and 'nut'")
    if outputs["k"] != field("k") or outputs["nut"] != field("nut"):
        raise ValueError("kEqnFjord outputs must bind to fields 'k' and 'nut'")
    if not all(isinstance(value, FieldExpression) for value in traced.values()):
        raise TypeError("native function outputs must be symbolic field expressions")
    if _uses_patch(traced.values()):
        raise ValueError("Field.transport does not support patch fields")

    barriers = {
        node
        for expression in traced.values()
        for node in _walk(expression)
        if node.operation == "transport"
    }
    if len(barriers) != 1:
        raise ValueError("native function must contain exactly one transport solve")
    barrier = barriers.pop()
    transported, diffusivity, source, implicit_sink, lower_bound = barrier.arguments
    if transported.operation != "field" or transported.field_name != "k":
        raise ValueError("kEqnFjord transport must advance field 'k'")
    if lower_bound.operation != "constant" or lower_bound.field_name is None:
        raise ValueError("transport lower_bound must be a finite scalar constant")
    bound = float(lower_bound.field_name)
    if bound < 0.0:
        raise ValueError("kEqnFjord transport lower_bound must not be negative")
    if any(
        expression.contains_transport
        for expression in (diffusivity, source, implicit_sink)
    ):
        raise ValueError("transport coefficients must not contain another solve")
    if any(
        _uses_outside(expression, transported, barrier)
        for name, expression in traced.items()
        if name != "k"
    ):
        raise ValueError(
            "native transport outputs cannot mix pre-solve and post-solve 'k'"
        )
    replacement = field("k")
    lowered_outputs = {
        name: _replace(expression, barrier, replacement)
        for name, expression in traced.items()
    }
    if lowered_outputs["k"] != replacement:
        raise ValueError("native function output 'k' must be the transport result")
    return NativeTransportPlan(
        field=transported,
        diffusivity=diffusivity,
        source=source,
        implicit_sink=implicit_sink,
        lower_bound=bound,
        outputs=lowered_outputs,
    )


def compile_native_program(
    function: Callable[..., object],
    inputs: Mapping[str, FieldExpression],
    outputs: Mapping[str, FieldExpression],
) -> NativeFieldPlan | None:
    """Compile a symbolic function when its complete graph is solver-native."""

    if _uses_patch((*inputs.values(), *outputs.values())):
        return None
    if requires_native_transport(function):
        return compile_native_transport(function, inputs, outputs)
    try:
        result = function(**inputs)
    except Exception:
        return None
    if isinstance(result, Mapping):
        traced = dict(result)
    elif len(outputs) == 1:
        traced = {next(iter(outputs)): result}
    else:
        return None
    if set(traced) != set(outputs) or not all(
        isinstance(value, FieldExpression) for value in traced.values()
    ):
        return None
    if _uses_patch(traced.values()):
        return None
    if any(value.contains_transport for value in traced.values()):
        return None
    return NativeAlgebraPlan(traced)


__all__ = [
    "NativeAlgebraPlan",
    "NativeFieldPlan",
    "NativeTransportPlan",
    "compile_native_program",
    "compile_native_transport",
    "requires_native_transport",
]
