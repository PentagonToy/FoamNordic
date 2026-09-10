from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from foamnordic.contracts import adapter_contract
from foamnordic.models.artifact import FnomArtifact, TensorContract
import foamnordic as fno


class AdapterContractTests(unittest.TestCase):
    def test_k_equation_contract_owns_internal_output_layouts(self) -> None:
        contract = adapter_contract("kEqnFjord")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.category, "les_model")
        self.assertEqual(contract.inputs["grad(U)"].kind, "tensor")
        self.assertEqual(contract.inputs["grad(U)"].physical_shape, (3, 3))
        self.assertEqual(
            tuple(contract.outputs),
            ("nut", "kProduction", "kDissipationCoeff"),
        )
        self.assertTrue(
            all(layout.kind == "scalar" for layout in contract.outputs.values())
        )

    def test_adapter_names_are_case_sensitive(self) -> None:
        self.assertIsNone(adapter_contract("KEqnFjord"))
        self.assertIsNone(adapter_contract("keqnfjord"))

    def test_contract_mappings_are_immutable(self) -> None:
        contract = adapter_contract("nutFjord")

        self.assertIsNotNone(contract)
        assert contract is not None
        with self.assertRaises(TypeError):
            contract.outputs["later"] = contract.outputs["nut"]  # type: ignore[index]

    @patch("foamnordic.contracts.bindings.load")
    def test_model_closure_infers_builtin_semantic_bindings(self, load) -> None:
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

        self.assertEqual(tuple(closure.inputs), ("velocity_grad", "filter_width"))
        self.assertEqual(closure.inputs["velocity_grad"].canonical, "grad(U)")
        self.assertEqual(closure.inputs["filter_width"].canonical, "delta")
        self.assertEqual(tuple(closure.outputs), ("eddy_viscosity",))
        self.assertEqual(closure.outputs["eddy_viscosity"].canonical, "nut")

    @patch("foamnordic.contracts.bindings.load")
    def test_inference_rejects_incompatible_tensor_width(self, load) -> None:
        artifact = FnomArtifact(
            path=Path("model.fnom"),
            name="model", format="onnx",
            inputs=(TensorContract("velocity_grad", 6, "float32"),),
            outputs=(TensorContract("eddy_viscosity", 1, "float32"),),
            runtime=None, input_scaler=None, output_scaler=None, bundled=True,
            payload_offset=64, payload_size=1,
        )
        load.return_value = artifact
        with self.assertRaisesRegex(ValueError, "declare inputs explicitly"):
            fno.Closure(
                name="nutFjord",
                operator=fno.Operator.model("model.fnom"),
            )

    @patch("foamnordic.contracts.bindings.load")
    def test_inference_rejects_incomplete_builtin_contract(self, load) -> None:
        load.return_value = FnomArtifact(
            path=Path("model.fnom"),
            name="model", format="onnx",
            inputs=(TensorContract("velocity_grad", 9, "float32"),),
            outputs=(TensorContract("eddy_viscosity", 1, "float32"),),
            runtime=None, input_scaler=None, output_scaler=None, bundled=True,
            payload_offset=64, payload_size=1,
        )
        with self.assertRaisesRegex(ValueError, "missing required adapter port"):
            fno.Closure(
                name="nutFjord",
                operator=fno.Operator.model("model.fnom"),
            )

    def test_unknown_adapter_requires_explicit_bindings(self) -> None:
        with self.assertRaisesRegex(ValueError, "no built-in OpenFOAM adapter"):
            fno.Closure(
                name="customClosure",
                operator=fno.Operator.model("model.fnom"),
            )

    @patch("foamnordic.contracts.bindings.load")
    def test_transform_infers_common_stored_fields(self, load) -> None:
        load.return_value = FnomArtifact(
            path=Path("model.fnom"),
            name="model", format="onnx",
            inputs=(
                TensorContract("pressure", 1, "float32"),
                TensorContract("eddy_viscosity", 1, "float32"),
            ),
            outputs=(
                TensorContract("predicted_velocity", 3, "float32"),
                TensorContract("temperature", 1, "float32"),
            ),
            runtime=None, input_scaler=None, output_scaler=None, bundled=True,
            payload_offset=64, payload_size=1,
        )

        transform = fno.Transform(
            name="correctFields",
            operator=fno.Operator.model("model.fnom"),
        )

        self.assertEqual(
            {name: value.canonical for name, value in transform.inputs.items()},
            {"pressure": "p", "eddy_viscosity": "nut"},
        )
        self.assertEqual(
            {name: value.canonical for name, value in transform.outputs.items()},
            {"predicted_velocity": "U", "temperature": "T"},
        )

    @patch("foamnordic.contracts.bindings.load")
    def test_transform_rejects_unknown_semantics(self, load) -> None:
        load.return_value = FnomArtifact(
            path=Path("model.fnom"),
            name="model", format="onnx",
            inputs=(TensorContract("mystery", 1, "float32"),),
            outputs=(TensorContract("temperature", 1, "float32"),),
            runtime=None, input_scaler=None, output_scaler=None, bundled=True,
            payload_offset=64, payload_size=1,
        )
        with self.assertRaisesRegex(ValueError, "declare inputs explicitly"):
            fno.Transform(
                name="unknownInput",
                operator=fno.Operator.model("model.fnom"),
            )


if __name__ == "__main__":
    unittest.main()
