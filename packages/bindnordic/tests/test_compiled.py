"""Exercise an installed BindNordic without importing FoamNordic."""

import ctypes
import os
import tempfile

import numpy as np
from bindnordic.models.compiled import generate_source
from bindnordic.models.compiler import compile_source
from bindnordic.models.forest import encode
from sklearn.ensemble import ExtraTreesRegressor


def test_predictions_and_cache(mocker):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 3))
    model = ExtraTreesRegressor(n_estimators=4, max_depth=4, random_state=42)
    model.fit(x, x[:, 0] * x[:, 1] + x[:, 2])
    source = generate_source(model, input_width=3, output_width=1).encode()
    assert encode(model, 3).startswith(b"FNOFOR1\0")
    with tempfile.TemporaryDirectory() as cache:
        _mock_patch_1 = mocker.patch.dict(
            os.environ, {"BINDNORDIC_COMPILED_CACHE": cache}
        )
        try:
            path, hit, _ = compile_source(source)
            assert not hit
            assert compile_source(source)[1]
            library = ctypes.CDLL(str(path))
            predict = library.bindnordic_predict
            pointer = ctypes.POINTER(ctypes.c_double)
            predict.argtypes = [
                pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                pointer,
                ctypes.c_size_t,
            ]
            predict.restype = ctypes.c_int
            output = np.empty(len(x))
            code = predict(
                x.ctypes.data_as(pointer), len(x), 3, output.ctypes.data_as(pointer), 1
            )
            assert code == 0
            np.testing.assert_allclose(output, model.predict(x), rtol=1e-13, atol=1e-13)
        finally:
            mocker.stop(_mock_patch_1)
