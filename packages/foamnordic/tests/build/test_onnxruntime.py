from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from foamnordic.build.onnxruntime import resolve


def test_explicit_root_requires_api_28(mocker) -> None:
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
            "libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so"
        )
        (library / library_name).touch()
        _mock_patch_1 = mocker.patch.dict(
            os.environ, {"FOAMNORDIC_ONNX_RUNTIME_ROOT": str(root)}
        )
        try:
            selected = resolve(download=False)
        finally:
            mocker.stop(_mock_patch_1)
        assert selected.root == root.resolve()
        assert selected.source == "environment"

        (include / "onnxruntime_c_api.h").write_text(
            "#define ORT_API_VERSION 29\n", encoding="utf-8"
        )
        _mock_patch_2 = mocker.patch.dict(
            os.environ, {"FOAMNORDIC_ONNX_RUNTIME_ROOT": str(root)}
        )
        try:
            with (
                pytest.raises(RuntimeError, match="API 28"),
            ):
                resolve(download=False)
        finally:
            mocker.stop(_mock_patch_2)
