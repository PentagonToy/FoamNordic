"""Binding-neutral OpenFOAM field expressions."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_nonempty


@dataclass(frozen=True, slots=True)
class FieldExpression:
    """A declarative OpenFOAM field or native field operation."""

    operation: str
    field_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", require_nonempty(self.operation, "operation"))
        if self.operation not in {"field", "grad", "filter_width"}:
            raise ValueError(f"unsupported field operation: {self.operation}")
        if self.operation == "filter_width" and self.field_name is not None:
            raise ValueError("filter_width does not accept a field name")
        if self.operation != "filter_width" and self.field_name is None:
            raise ValueError(f"{self.operation} requires a field name")
        if self.field_name is not None:
            object.__setattr__(
                self,
                "field_name",
                require_nonempty(self.field_name, "field_name"),
            )

    def to_plan(self) -> dict[str, str]:
        value = {"operation": self.operation}
        if self.field_name is not None:
            value["field"] = self.field_name
        return value


def field(name: str) -> FieldExpression:
    """Bind a logical tensor to an OpenFOAM field."""

    return FieldExpression("field", name)


def grad(name: str) -> FieldExpression:
    """Request the native gradient of an OpenFOAM field."""

    return FieldExpression("grad", name)


def filter_width() -> FieldExpression:
    """Request the LES filter-width expression."""

    return FieldExpression("filter_width")
