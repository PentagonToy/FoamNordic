from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import foamnordic as fno
from foamnordic._native_plan import available as native_available


class ExportTests(unittest.TestCase):
    def test_verbose_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "boolean"):
            fno.export.onnx(
                b"onnx",
                path="model.fnom",
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                verbose="yes",
            )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_existing_onnx_payload_exports_native_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fno.export.onnx(
                b"onnx-payload",
                path=root / "reaction-rate.fnom",
                name="reaction-rate",
                inputs={
                    "c_tilde": fno.Tensor.scalar(),
                    "c_var": fno.Tensor.scalar(),
                    "T_tilde": fno.Tensor.scalar(),
                },
                outputs={"omega_c": fno.Tensor.scalar()},
            )
            self.assertEqual(manifest.read_bytes()[:8], b"FNOMAN1\0")
            self.assertEqual(
                (root / "reaction-rate.onnx").read_bytes(), b"onnx-payload"
            )
            self.assertLess(manifest.stat().st_size, 1024)

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_path_backed_payload_is_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large-voting-regressor.onnx"
            source.write_bytes(b"model" * 1024)
            manifest = fno.export.onnx(
                source,
                path=root / "voting.fnom",
                inputs={"features": fno.Tensor.vector(components=3)},
                outputs={"prediction": fno.Tensor.scalar()},
            )
            self.assertEqual(
                manifest.with_suffix(".onnx").read_bytes(), source.read_bytes()
            )

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_verbose_true_displays_export_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "foamnordic.export._display_export"
        ) as display:
            fno.export.onnx(
                b"onnx-payload",
                path=Path(directory) / "model.fnom",
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                verbose=True,
            )
        display.assert_called_once()


if __name__ == "__main__":
    unittest.main()
