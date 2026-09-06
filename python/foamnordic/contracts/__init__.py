"""Declarative contracts for built-in FoamNordic integrations."""

from .adapters import AdapterContract, adapter_contract
from .bindings import infer_model_bindings, infer_transform_bindings

__all__ = [
    "AdapterContract",
    "adapter_contract",
    "infer_model_bindings",
    "infer_transform_bindings",
]
