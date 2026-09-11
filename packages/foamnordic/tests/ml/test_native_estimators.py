import tempfile
from pathlib import Path

import numpy as np
from foamnordic.build.compiler import compile_source

import foamnordic as fno


def test_forest_data_and_compiled_native_match_sklearn():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    rng = np.random.default_rng(12)
    x = rng.normal(size=(512, 3))
    y = x[:, 0] * x[:, 1] + x[:, 2]
    for constructor in (ExtraTreesRegressor, RandomForestRegressor):
        model = constructor(n_estimators=7, max_depth=5, random_state=12).fit(x, y)
        for backend in ("forest", "compiled"):
            with tempfile.TemporaryDirectory() as tmp:
                path = fno.export.sklearn(
                    model,
                    path=Path(tmp) / "model.fnom",
                    inputs={"x": fno.Tensor.vector(components=3)},
                    outputs={"y": fno.Tensor.scalar()},
                    backend=backend,
                )
                library = ""
                if backend == "compiled":
                    library = str(
                        compile_source(fno._native.read_model_payload(str(path)))[0]
                    )
                for rows in (0, 1, 511, 32768):
                    values = np.ascontiguousarray(rng.normal(size=(rows, 3)))
                    actual = fno._native._evaluate_native_model(
                        str(path), library, values.tobytes(), rows, 3, 3
                    )
                    predicted = np.frombuffer(actual, dtype=np.float64)
                    expected = model.predict(values) if rows else np.empty(0)
                    np.testing.assert_allclose(
                        predicted, expected, rtol=1e-13, atol=1e-13
                    )
