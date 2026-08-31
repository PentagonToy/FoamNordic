"""Validated access to built-in OpenFOAM adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

from ..core.layout import FieldLayout, field_layout


_SCHEMA = "foamnordic.openfoam-adapters/v1"


@dataclass(frozen=True, slots=True)
class AdapterContract:
    """Immutable field contract owned by one built-in OpenFOAM adapter."""

    name: str
    category: str
    inputs: Mapping[str, FieldLayout]
    outputs: Mapping[str, FieldLayout]


def _ports(value: object, label: str) -> Mapping[str, FieldLayout]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{label} must be a non-empty mapping")
    ports: dict[str, FieldLayout] = {}
    for name, declaration in value.items():
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"{label} contains an invalid field name")
        if not isinstance(declaration, dict) or set(declaration) != {"kind"}:
            raise RuntimeError(f"{label}.{name} must declare exactly one kind")
        kind = declaration["kind"]
        if not isinstance(kind, str):
            raise RuntimeError(f"{label}.{name}.kind must be a string")
        try:
            ports[name] = field_layout(kind)
        except ValueError as error:
            raise RuntimeError(f"{label}.{name}: {error}") from error
    return MappingProxyType(ports)


@lru_cache(maxsize=1)
def _contracts() -> Mapping[str, AdapterContract]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("FoamNordic adapter contracts require PyYAML") from error

    path = files(__package__).joinpath("openfoam_adapters.yaml")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
        raise RuntimeError(f"OpenFOAM adapter contracts must use schema {_SCHEMA}")
    raw_adapters = document.get("adapters")
    if not isinstance(raw_adapters, dict) or not raw_adapters:
        raise RuntimeError("OpenFOAM adapter contracts must define adapters")

    contracts: dict[str, AdapterContract] = {}
    for name, declaration in raw_adapters.items():
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("OpenFOAM adapter contract has an invalid name")
        if not isinstance(declaration, dict):
            raise RuntimeError(f"adapter {name} must be a mapping")
        if set(declaration) != {"category", "inputs", "outputs"}:
            raise RuntimeError(
                f"adapter {name} must declare category, inputs, and outputs"
            )
        category = declaration["category"]
        if not isinstance(category, str) or not category.strip():
            raise RuntimeError(f"adapter {name} category must be a non-empty string")
        contracts[name] = AdapterContract(
            name=name,
            category=category,
            inputs=_ports(declaration["inputs"], f"{name}.inputs"),
            outputs=_ports(declaration["outputs"], f"{name}.outputs"),
        )
    return MappingProxyType(contracts)


def adapter_contract(name: str) -> AdapterContract | None:
    """Return a strict case-sensitive built-in adapter contract, if known."""

    return _contracts().get(name)
