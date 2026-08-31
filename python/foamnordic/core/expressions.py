"""Binding-neutral OpenFOAM field expressions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .validation import require_nonempty


_ARITIES = {
    "grad": frozenset({1}),
    "div": frozenset({1, 2}),
    "laplacian": frozenset({1, 2}),
    "curl": frozenset({1}),
    "dot": frozenset({2}),
    "ddot": frozenset({2}),
    "mag": frozenset({1}),
    "symm": frozenset({1}),
    "dev": frozenset({1}),
}
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]*$")
_FIELD_PATTERN = re.compile(r"^[A-Za-z_*][A-Za-z0-9_.*:]*$")


def _field_name(value: str) -> str:
    name = require_nonempty(value, "field_name")
    if _FIELD_NAME.fullmatch(name) is None:
        raise ValueError(
            "field_name must be an OpenFOAM word containing letters, digits, "
            "underscore, dot, or colon"
        )
    return name


@dataclass(frozen=True, slots=True)
class FieldExpression:
    """An immutable OpenFOAM field or finite-volume operation tree.

    ``field_name`` preserves the original compact representation for direct
    unary expressions such as ``grad(U)``. Nested and binary expressions use
    ``arguments``. Both forms have one canonical OpenFOAM expression.
    """

    operation: str
    field_name: str | None = None
    arguments: tuple["FieldExpression", ...] = ()

    def __post_init__(self) -> None:
        operation = require_nonempty(self.operation, "operation")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "arguments", tuple(self.arguments))

        if operation == "field":
            if self.field_name is None or self.arguments:
                raise ValueError("field requires one field name")
            object.__setattr__(
                self,
                "field_name",
                _field_name(self.field_name),
            )
            return

        if operation == "filter_width":
            if self.field_name is not None or self.arguments:
                raise ValueError("filter_width does not accept arguments")
            return

        if operation not in _ARITIES:
            raise ValueError(f"unsupported field operation: {operation}")
        if self.field_name is not None:
            if self.arguments or 1 not in _ARITIES[operation]:
                raise ValueError(
                    f"{operation} compact field form requires one argument"
                )
            object.__setattr__(
                self,
                "field_name",
                _field_name(self.field_name),
            )
            return
        if len(self.arguments) not in _ARITIES[operation]:
            expected = " or ".join(
                str(value) for value in sorted(_ARITIES[operation])
            )
            raise ValueError(
                f"{operation} requires {expected} argument(s), "
                f"got {len(self.arguments)}"
            )
        if not all(
            isinstance(value, FieldExpression) for value in self.arguments
        ):
            raise TypeError(
                "field expression arguments must be FieldExpression values"
            )

    @property
    def canonical(self) -> str:
        """Return the canonical expression consumed by OpenFOAM."""

        if self.operation == "field":
            assert self.field_name is not None
            return self.field_name
        if self.operation == "filter_width":
            return "delta"
        if self.field_name is not None:
            return f"{self.operation}({self.field_name})"
        return (
            f"{self.operation}("
            + ",".join(argument.canonical for argument in self.arguments)
            + ")"
        )

    @property
    def derived(self) -> bool:
        return self.operation not in {"field", "filter_width"}

    def to_plan(self) -> dict[str, object]:
        value: dict[str, object] = {"operation": self.operation}
        if self.field_name is not None:
            value["field"] = self.field_name
        if self.arguments:
            value["arguments"] = [
                argument.to_plan() for argument in self.arguments
            ]
        return value


@dataclass(frozen=True, slots=True)
class FieldSelection:
    """Select a family of stored OpenFOAM fields by name pattern.

    A selection is metadata, not a finite-volume expression. It is intended
    for structured outputs such as a flamelet manifold's species family.
    Expansion occurs against the solver's object registry before launch; a
    wildcard that matches no fields is an error at that boundary.
    """

    pattern: str

    def __post_init__(self) -> None:
        pattern = require_nonempty(self.pattern, "field pattern")
        if _FIELD_PATTERN.fullmatch(pattern) is None:
            raise ValueError(
                "field pattern must contain only OpenFOAM word characters "
                "and '*' wildcards"
            )
        if "*" not in pattern:
            raise ValueError("fields() requires at least one '*' wildcard")
        object.__setattr__(self, "pattern", pattern)

    def to_plan(self) -> dict[str, object]:
        return {"selection": "fields", "pattern": self.pattern}


def field(name: str) -> FieldExpression:
    """Bind a logical tensor to an OpenFOAM field."""

    return FieldExpression("field", name)


def fields(pattern: str) -> FieldSelection:
    """Select multiple stored OpenFOAM fields using one name wildcard."""

    return FieldSelection(pattern)


def _argument(value: str | FieldExpression) -> FieldExpression:
    if isinstance(value, FieldExpression):
        return value
    if isinstance(value, str):
        return field(value)
    raise TypeError(
        "OpenFOAM operation arguments must be field names or expressions"
    )


def operation(
    name: str, *arguments: str | FieldExpression
) -> FieldExpression:
    """Build one validated OpenFOAM operation expression."""

    operator = require_nonempty(name, "operation")
    normalized = tuple(_argument(value) for value in arguments)
    if len(normalized) == 1 and normalized[0].operation == "field":
        return FieldExpression(operator, normalized[0].field_name)
    return FieldExpression(operator, arguments=normalized)


def grad(value: str | FieldExpression) -> FieldExpression:
    """Request the native gradient of an OpenFOAM field or expression."""

    return operation("grad", value)


def filter_width() -> FieldExpression:
    """Request the LES filter-width expression."""

    return FieldExpression("filter_width")
