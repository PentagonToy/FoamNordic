from __future__ import annotations

import unittest

from foamnordic.contracts import adapter_contract


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


if __name__ == "__main__":
    unittest.main()
