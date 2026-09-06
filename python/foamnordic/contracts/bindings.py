"""Conservative FNOM-to-OpenFOAM binding resolution."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping

from ..core.expressions import FieldExpression, field, filter_width, grad
from ..core.layout import FieldLayout
from ..core.paths import PathInput
from ..models.artifact import TensorContract, load
from .adapters import AdapterContract, adapter_contract


_ALIASES = {
    "velocitygrad": "grad(U)",
    "velocitygradient": "grad(U)",
    "gradu": "grad(U)",
    "filterwidth": "delta",
    "eddyviscosity": "nut",
    "turbulentviscosity": "nut",
    "kproduction": "kProduction",
    "kdissipationcoeff": "kDissipationCoeff",
}

_STORED_FIELDS = {
    "u": ("U", 3),
    "velocity": ("U", 3),
    "predictedvelocity": ("U", 3),
    "correctedvelocity": ("U", 3),
    "updatedvelocity": ("U", 3),
    "p": ("p", 1),
    "pressure": ("p", 1),
    "predictedpressure": ("p", 1),
    "correctedpressure": ("p", 1),
    "t": ("T", 1),
    "temperature": ("T", 1),
    "predictedtemperature": ("T", 1),
    "correctedtemperature": ("T", 1),
    "k": ("k", 1),
    "turbulentkineticenergy": ("k", 1),
    "nut": ("nut", 1),
    "eddyviscosity": ("nut", 1),
    "turbulentviscosity": ("nut", 1),
}


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _expression(name: str) -> FieldExpression:
    if name == "delta":
        return filter_width()
    if name == "grad(U)":
        return grad("U")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:]*", name):
        return field(name)
    raise ValueError(
        f"built-in adapter field {name!r} has no supported expression binding"
    )


def _resolve_ports(
    ports: tuple[TensorContract, ...],
    expected: Mapping[str, FieldLayout],
    *,
    label: str,
) -> Mapping[str, FieldExpression]:
    available = dict(expected)
    resolved: dict[str, FieldExpression] = {}
    for port in ports:
        alias = _ALIASES.get(_token(port.name))
        candidates = [
            name
            for name in available
            if _token(name) == _token(port.name) or name == alias
        ]
        candidates = [
            name
            for name in candidates
            if available[name].transport_width == port.components
        ]
        if len(candidates) != 1:
            detail = "no compatible field" if not candidates else "ambiguous fields"
            raise ValueError(
                f"cannot infer {label} binding for FNOM port {port.name!r}: "
                f"{detail}; declare {label} explicitly"
            )
        selected = candidates[0]
        resolved[port.name] = _expression(selected)
        del available[selected]
    if available:
        raise ValueError(
            f"cannot infer {label}: FNOM is missing required adapter port(s) "
            f"{tuple(available)!r}; declare {label} explicitly"
        )
    return MappingProxyType(resolved)


def infer_model_bindings(
    name: str,
    artifact: PathInput,
    *,
    inputs: Mapping[str, FieldExpression] | None,
    outputs: Mapping[str, FieldExpression] | None,
) -> tuple[Mapping[str, FieldExpression], Mapping[str, FieldExpression]]:
    """Fill omitted bindings from one strict built-in adapter contract."""

    if inputs is not None and outputs is not None:
        return inputs, outputs
    contract: AdapterContract | None = adapter_contract(name)
    if contract is None:
        missing = "inputs and outputs" if inputs is None and outputs is None else (
            "inputs" if inputs is None else "outputs"
        )
        raise ValueError(
            f"cannot infer {missing} for {name!r}; no built-in OpenFOAM "
            "adapter contract exists"
        )
    manifest = load(artifact)
    return (
        inputs
        if inputs is not None
        else _resolve_ports(manifest.inputs, contract.inputs, label="inputs"),
        outputs
        if outputs is not None
        else _resolve_ports(manifest.outputs, contract.outputs, label="outputs"),
    )


def _resolve_stored_ports(
    ports: tuple[TensorContract, ...], *, label: str
) -> Mapping[str, FieldExpression]:
    resolved: dict[str, FieldExpression] = {}
    for port in ports:
        selected = _STORED_FIELDS.get(_token(port.name))
        if selected is None or selected[1] != port.components:
            raise ValueError(
                f"cannot infer {label} binding for FNOM port {port.name!r}; "
                f"declare {label} explicitly"
            )
        resolved[port.name] = field(selected[0])
    return MappingProxyType(resolved)


def infer_transform_bindings(
    artifact: PathInput,
    *,
    inputs: Mapping[str, FieldExpression] | None,
    outputs: Mapping[str, FieldExpression] | None,
) -> tuple[Mapping[str, FieldExpression], Mapping[str, FieldExpression]]:
    """Fill omitted direct stored-field bindings from an FNOM contract."""

    if inputs is not None and outputs is not None:
        return inputs, outputs
    manifest = load(artifact)
    return (
        inputs
        if inputs is not None
        else _resolve_stored_ports(manifest.inputs, label="inputs"),
        outputs
        if outputs is not None
        else _resolve_stored_ports(manifest.outputs, label="outputs"),
    )
