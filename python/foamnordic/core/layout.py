"""Physical and transport layouts for OpenFOAM field values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FieldLayout:
    """Describe one value at the Python/OpenFOAM transport boundary."""

    kind: str
    physical_shape: tuple[int, ...]
    transport_width: int

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("field layout kind must not be empty")
        if self.transport_width <= 0:
            raise ValueError("field layout transport width must be positive")
        if any(value <= 0 for value in self.physical_shape):
            raise ValueError("field layout dimensions must be positive")

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "physical_shape": list(self.physical_shape),
            "transport_width": self.transport_width,
        }

    @classmethod
    def from_plan(cls, value: Mapping[str, object]) -> "FieldLayout":
        return cls(
            kind=str(value["kind"]),
            physical_shape=tuple(
                int(item) for item in value.get("physical_shape", ())
            ),
            transport_width=int(value["transport_width"]),
        )

    def unpack(self, values, rows: int):
        """Decode transport storage into the Python physical shape."""

        import numpy as np

        packed = np.asarray(values).reshape(rows, self.transport_width)
        if self.kind == "scalar":
            return packed[:, 0]
        if self.kind == "symm_tensor":
            xx, xy, xz, yy, yz, zz = np.moveaxis(packed, -1, 0)
            physical = np.empty((rows, 3, 3), dtype=packed.dtype)
            physical[:, 0, 0] = xx
            physical[:, 0, 1] = physical[:, 1, 0] = xy
            physical[:, 0, 2] = physical[:, 2, 0] = xz
            physical[:, 1, 1] = yy
            physical[:, 1, 2] = physical[:, 2, 1] = yz
            physical[:, 2, 2] = zz
            return physical
        return packed.reshape(rows, *self.physical_shape)

    def pack(self, values, rows: int):
        """Encode a Python physical value into OpenFOAM transport storage."""

        import numpy as np

        array = np.asarray(values)
        if self.kind == "symm_tensor" and array.shape == (rows, 3, 3):
            array = np.column_stack(
                (
                    array[:, 0, 0],
                    array[:, 0, 1],
                    array[:, 0, 2],
                    array[:, 1, 1],
                    array[:, 1, 2],
                    array[:, 2, 2],
                )
            )
        try:
            return array.reshape(rows, self.transport_width)
        except ValueError as error:
            physical = (rows, *self.physical_shape)
            transport = (rows, self.transport_width)
            raise ValueError(
                f"Operator.function {self.kind} output must have shape "
                f"{physical} or {transport}"
            ) from error


_LAYOUTS = {
    "scalar": FieldLayout("scalar", (), 1),
    "vector": FieldLayout("vector", (3,), 3),
    "spherical_tensor": FieldLayout("spherical_tensor", (1,), 1),
    # OpenFOAM transports symmetric tensors in six-component compact form.
    # Python receives the complete matrix so all tensor algebra uses the same
    # physical representation.
    "symm_tensor": FieldLayout("symm_tensor", (3, 3), 6),
    "tensor": FieldLayout("tensor", (3, 3), 9),
}


def field_layout(kind: str) -> FieldLayout:
    try:
        return _LAYOUTS[kind]
    except KeyError:
        raise ValueError(f"unsupported OpenFOAM value kind: {kind}") from None


__all__ = ["FieldLayout", "field_layout"]
