from __future__ import annotations

from io import BytesIO

import joblib
import numpy as np
import pytest
from bindnordic.models.runtime import joblib_evaluator, packed_result


class _Predictor:
    def predict(self, values):
        return values[:, 0] + 2.0 * values[:, 1]


class _NamedPredictor:
    def predict(self, values):
        return values["first"] + 2.0 * values["second"]


def test_joblib_evaluator_is_solver_independent():
    payload = BytesIO()
    joblib.dump(_Predictor(), payload)
    payload.seek(0)
    evaluator = joblib_evaluator(payload, (("result", 1),), threads=1)
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    result = evaluator(values.data, 2, 2, "float64", 7, 0.25, 3)

    np.testing.assert_array_equal(result, [[5.0], [11.0]])


@pytest.mark.parametrize("input_format", ["pandas", "polars"])
def test_joblib_evaluator_builds_named_tabular_input(input_format):
    pytest.importorskip(input_format)
    payload = BytesIO()
    joblib.dump(_NamedPredictor(), payload)
    payload.seek(0)
    evaluator = joblib_evaluator(
        payload,
        (("result", 1),),
        threads=1,
        input_names=("first", "second"),
        input_format=input_format,
    )
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    result = evaluator(values.data, 2, 2, "float64", 7, 0.25, 3)

    np.testing.assert_array_equal(result, [[5.0], [11.0]])


def test_packed_result_preserves_declared_output_order():
    result = packed_result(
        {"second": [3.0, 4.0], "first": [1.0, 2.0]},
        (("first", 1), ("second", 1)),
        2,
        "float64",
    )

    np.testing.assert_array_equal(result, [[1.0, 3.0], [2.0, 4.0]])
