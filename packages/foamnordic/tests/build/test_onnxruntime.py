from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from foamnordic.build.onnxruntime import resolve


class OnnxRuntimeResolverTests(unittest.TestCase):
    def test_explicit_root_requires_api_28(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "include"
            library = root / "lib"
            include.mkdir()
            library.mkdir()
            (include / "onnxruntime_cxx_api.h").touch()
            (include / "onnxruntime_c_api.h").write_text(
                "#define ORT_API_VERSION 28\n", encoding="utf-8"
            )
            library_name = (
                "libonnxruntime.dylib"
                if sys.platform == "darwin"
                else "libonnxruntime.so"
            )
            (library / library_name).touch()

            with patch.dict(
                os.environ, {"FOAMNORDIC_ONNX_RUNTIME_ROOT": str(root)}
            ):
                selected = resolve(download=False)
            self.assertEqual(selected.root, root.resolve())
            self.assertEqual(selected.source, "environment")

            (include / "onnxruntime_c_api.h").write_text(
                "#define ORT_API_VERSION 29\n", encoding="utf-8"
            )
            with (
                patch.dict(
                    os.environ, {"FOAMNORDIC_ONNX_RUNTIME_ROOT": str(root)}
                ),
                self.assertRaisesRegex(RuntimeError, "API 28"),
            ):
                resolve(download=False)


if __name__ == "__main__":
    unittest.main()
