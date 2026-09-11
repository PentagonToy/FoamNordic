from __future__ import annotations

import os

import pytest
from bindnordic.models.runtime import (
    _configure_estimator_threads,
)
from bindnordic.models.runtime import (
    configure_model_threads as _configure_model_threads,
)
from foamnordic.core.native_plan import local_cpu_budget
from foamnordic.execution.resident import (
    _parser,
    run_program,
)


class _Estimator:
    def __init__(self, n_jobs=None, estimators=()):
        self.n_jobs = n_jobs
        self.estimators_ = list(estimators)


def test_local_cpu_budget_respects_process_affinity(mocker) -> None:
    affinity = mocker.patch(
        "foamnordic.core.native_plan.os.sched_getaffinity",
        create=True,
    )
    try:
        affinity.return_value = {2, 4, 6}
        assert local_cpu_budget() == 3
    finally:
        mocker.stop(affinity)


def test_model_thread_budget_caps_native_runtime_environment(mocker) -> None:
    environment = {
        "OMP_NUM_THREADS": "64",
        "MKL_NUM_THREADS": "2",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    _mock_patch_2 = mocker.patch.dict(os.environ, environment, clear=True)
    try:
        assert _configure_model_threads(8) == 8
        assert os.environ["OMP_NUM_THREADS"] == "8"
        assert os.environ["MKL_NUM_THREADS"] == "8"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
        assert "xla_cpu_multi_thread_eigen=true" in os.environ["XLA_FLAGS"]
        assert "intra_op_parallelism_threads=8" in os.environ["XLA_FLAGS"]
    finally:
        mocker.stop(_mock_patch_2)


def test_model_thread_budget_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        _configure_model_threads(0)


def test_fitted_estimator_graph_receives_the_budget() -> None:
    nested = _Estimator(n_jobs=1)
    model = _Estimator(n_jobs=-1, estimators=(nested,))
    _configure_estimator_threads(model, 4)
    assert model.n_jobs == 4
    assert nested.n_jobs == 4


def test_resident_cli_accepts_threads() -> None:
    arguments = _parser().parse_args(
        ("unix:///tmp/model.sock", "model.fnom", "--threads", "12")
    )
    assert arguments.threads == 12


@pytest.mark.parametrize("model_format", ("onnx", "table"))
def test_grouped_native_backend_uses_native_worker(model_format, mocker) -> None:
    manifest = {
        "format": model_format,
        "runtime": model_format,
        "inputs": [("value", 1, "float64")],
        "outputs": [("value", 1, "float64")],
    }
    mocker.patch(
        "foamnordic.execution.resident._manifest",
        return_value=manifest,
    )
    worker = mocker.patch(
        "foamnordic.execution.resident._native.run_native_worker"
    )
    capacity = object()

    assert run_program(
        [
            "unix:///tmp/model.sock",
            "model.fnom",
            "--threads",
            "4",
            "--ready-file",
            "/tmp/model.ready",
        ],
        capacity,
        configure_threads=False,
    ) == 0

    worker.assert_called_once()
    assert worker.call_args.args[-1] is capacity
