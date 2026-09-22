from __future__ import annotations

from pathlib import Path

import pytest
from foamnordic.contracts import adapter_contract
from foamnordic.models.artifact import FnomArtifact, TensorContract

import foamnordic as fno


def test_k_equation_contract_owns_internal_output_layouts() -> None:
    contract = adapter_contract("kEqnFjord")

    assert contract is not None
    assert contract.category == "les_model"
    assert contract.native_execution == frozenset({"transport"})
    assert contract.inputs["grad(U)"].kind == "tensor"
    assert contract.inputs["grad(U)"].physical_shape == (3, 3)
    assert tuple(contract.outputs) == ("k", "nut")
    assert all(layout.kind == "scalar" for layout in contract.outputs.values())


def test_adapter_names_are_case_sensitive() -> None:
    assert adapter_contract("KEqnFjord") is None
    assert adapter_contract("keqnfjord") is None


def test_contract_mappings_are_immutable() -> None:
    contract = adapter_contract("nutFjord")

    assert contract is not None
    assert contract.native_execution == frozenset({"algebra"})
    with pytest.raises(TypeError):
        contract.outputs["later"] = contract.outputs["nut"]  # type: ignore[index]


def test_progress_rate_contract_owns_logical_combustion_fields() -> None:
    contract = adapter_contract("progressRateFjord")

    assert contract is not None
    assert contract.category == "combustion_model"
    assert tuple(contract.inputs) == ("progress", "variance", "temperature")
    assert tuple(contract.outputs) == ("omega_c",)
    assert all(layout.kind == "scalar" for layout in contract.inputs.values())


def test_model_closure_infers_builtin_semantic_bindings(mocker) -> None:
    load = mocker.patch("foamnordic.contracts.bindings.load")
    load.return_value = FnomArtifact(
        path=Path("model.fnom"),
        name="model",
        format="onnx",
        inputs=(
            TensorContract("velocity_grad", 9, "float32"),
            TensorContract("filter_width", 1, "float32"),
        ),
        outputs=(TensorContract("eddy_viscosity", 1, "float32"),),
        runtime=None,
        input_scaler=None,
        output_scaler=None,
        bundled=True,
        payload_offset=64,
        payload_size=1,
    )

    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.model("model.fnom"),
    )

    assert tuple(closure.inputs) == ("velocity_grad", "filter_width")
    assert closure.inputs["velocity_grad"].canonical == "grad(U)"
    assert closure.inputs["filter_width"].canonical == "delta"
    assert tuple(closure.outputs) == ("eddy_viscosity",)
    assert closure.outputs["eddy_viscosity"].canonical == "nut"


def test_inference_rejects_incompatible_tensor_width(mocker) -> None:
    load = mocker.patch("foamnordic.contracts.bindings.load")
    artifact = FnomArtifact(
        path=Path("model.fnom"),
        name="model",
        format="onnx",
        inputs=(TensorContract("velocity_grad", 6, "float32"),),
        outputs=(TensorContract("eddy_viscosity", 1, "float32"),),
        runtime=None,
        input_scaler=None,
        output_scaler=None,
        bundled=True,
        payload_offset=64,
        payload_size=1,
    )
    load.return_value = artifact
    with pytest.raises(ValueError, match="declare inputs explicitly"):
        fno.Closure(
            name="nutFjord",
            operator=fno.Operator.model("model.fnom"),
        )


def test_inference_rejects_incomplete_builtin_contract(mocker) -> None:
    load = mocker.patch("foamnordic.contracts.bindings.load")
    load.return_value = FnomArtifact(
        path=Path("model.fnom"),
        name="model",
        format="onnx",
        inputs=(TensorContract("velocity_grad", 9, "float32"),),
        outputs=(TensorContract("eddy_viscosity", 1, "float32"),),
        runtime=None,
        input_scaler=None,
        output_scaler=None,
        bundled=True,
        payload_offset=64,
        payload_size=1,
    )
    with pytest.raises(ValueError, match="missing required adapter port"):
        fno.Closure(
            name="nutFjord",
            operator=fno.Operator.model("model.fnom"),
        )


def test_unknown_adapter_requires_explicit_bindings() -> None:
    with pytest.raises(ValueError, match="no built-in OpenFOAM adapter"):
        fno.Closure(
            name="customClosure",
            operator=fno.Operator.model("model.fnom"),
        )


def test_transform_infers_common_stored_fields(mocker) -> None:
    load = mocker.patch("foamnordic.contracts.bindings.load")
    load.return_value = FnomArtifact(
        path=Path("model.fnom"),
        name="model",
        format="onnx",
        inputs=(
            TensorContract("pressure", 1, "float32"),
            TensorContract("eddy_viscosity", 1, "float32"),
        ),
        outputs=(
            TensorContract("predicted_velocity", 3, "float32"),
            TensorContract("temperature", 1, "float32"),
        ),
        runtime=None,
        input_scaler=None,
        output_scaler=None,
        bundled=True,
        payload_offset=64,
        payload_size=1,
    )

    transform = fno.Transform(
        name="correctFields",
        operator=fno.Operator.model("model.fnom"),
    )

    assert {name: value.canonical for name, value in transform.inputs.items()} == {
        "pressure": "p",
        "eddy_viscosity": "nut",
    }
    assert {name: value.canonical for name, value in transform.outputs.items()} == {
        "predicted_velocity": "U",
        "temperature": "T",
    }


def test_transform_rejects_unknown_semantics(mocker) -> None:
    load = mocker.patch("foamnordic.contracts.bindings.load")
    load.return_value = FnomArtifact(
        path=Path("model.fnom"),
        name="model",
        format="onnx",
        inputs=(TensorContract("mystery", 1, "float32"),),
        outputs=(TensorContract("temperature", 1, "float32"),),
        runtime=None,
        input_scaler=None,
        output_scaler=None,
        bundled=True,
        payload_offset=64,
        payload_size=1,
    )
    with pytest.raises(ValueError, match="declare inputs explicitly"):
        fno.Transform(
            name="unknownInput",
            operator=fno.Operator.model("model.fnom"),
        )
