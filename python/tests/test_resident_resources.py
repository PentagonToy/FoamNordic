from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from foamnordic.execution.resident import (
    _configure_estimator_threads,
    _configure_model_threads,
    _parser,
)
from foamnordic.core.native_plan import local_cpu_budget


class _Estimator:
    def __init__(self, n_jobs=None, estimators=()):
        self.n_jobs = n_jobs
        self.estimators_ = list(estimators)


class ResidentResourceTests(unittest.TestCase):
    def test_local_cpu_budget_respects_process_affinity(self) -> None:
        with patch(
            "foamnordic.core.native_plan.os.sched_getaffinity",
            create=True,
        ) as affinity:
            affinity.return_value = {2, 4, 6}
            self.assertEqual(local_cpu_budget(), 3)

    def test_model_thread_budget_caps_native_runtime_environment(self) -> None:
        environment = {
            "OMP_NUM_THREADS": "64",
            "MKL_NUM_THREADS": "2",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(_configure_model_threads(8), 8)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "8")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "8")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")
            self.assertIn("xla_cpu_multi_thread_eigen=true", os.environ["XLA_FLAGS"])
            self.assertIn("intra_op_parallelism_threads=8", os.environ["XLA_FLAGS"])

    def test_model_thread_budget_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            _configure_model_threads(0)

    def test_fitted_estimator_graph_receives_the_budget(self) -> None:
        nested = _Estimator(n_jobs=1)
        model = _Estimator(n_jobs=-1, estimators=(nested,))
        _configure_estimator_threads(model, 4)
        self.assertEqual(model.n_jobs, 4)
        self.assertEqual(nested.n_jobs, 4)

    def test_resident_cli_accepts_threads(self) -> None:
        arguments = _parser().parse_args(
            ("unix:///tmp/model.sock", "model.fnom", "--threads", "12")
        )
        self.assertEqual(arguments.threads, 12)


if __name__ == "__main__":
    unittest.main()
