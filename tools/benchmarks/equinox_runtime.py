#!/usr/bin/env python3
"""Measure reproducible cold activation and warm Equinox inference."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import equinox as eqx
import jax
import numpy as np

import foamnordic as fno
from foamnordic.execution.resident import _equinox_evaluator


ROWS = 262_144
INPUTS = 10
COLD_REPETITIONS = 7
WARM_REPETITIONS = 20


def evaluate(manifest: Path):
    payload = BytesIO(fno._native.read_model_payload(str(manifest)))
    return _equinox_evaluator(payload, (("prediction", 1),), "float32", threads=1)


def values() -> np.ndarray:
    return np.linspace(-1.0, 1.0, ROWS * INPUTS, dtype=np.float32).reshape(
        ROWS, INPUTS
    )


def child(manifest: Path) -> None:
    features = values()
    started = time.perf_counter()
    evaluator = evaluate(manifest)
    result = evaluator(
        memoryview(features), ROWS, INPUTS, "float32", 0, 0.0
    )
    elapsed = time.perf_counter() - started
    print(json.dumps({"elapsed_s": elapsed, "checksum": float(result.sum())}))


def series(values_: list[float]) -> str:
    return (
        f"{statistics.median(values_):.6e} s "
        f"({min(values_):.6e} ~ {max(values_):.6e})"
    )


def benchmark() -> None:
    with tempfile.TemporaryDirectory(prefix="foamnordic-equinox-") as directory:
        manifest = Path(directory) / "equinox-mlp.fnom"
        model = eqx.nn.MLP(
            in_size=INPUTS,
            out_size=1,
            width_size=128,
            depth=3,
            key=jax.random.PRNGKey(42),
        )
        fno.export.equinox(
            model,
            path=manifest,
            inputs={
                "features": fno.Tensor.vector(
                    components=INPUTS, dtype="float32"
                )
            },
            outputs={"prediction": fno.Tensor.scalar(dtype="float32")},
        )

        cold = []
        checksums = []
        for _ in range(COLD_REPETITIONS):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(manifest)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            cold.append(float(result["elapsed_s"]))
            checksums.append(float(result["checksum"]))

        features = values()
        evaluator = evaluate(manifest)
        reference = evaluator(
            memoryview(features), ROWS, INPUTS, "float32", 0, 0.0
        )
        warm = []
        for exchange in range(1, WARM_REPETITIONS + 1):
            started = time.perf_counter()
            result = evaluator(
                memoryview(features), ROWS, INPUTS, "float32", exchange, 0.0
            )
            warm.append(time.perf_counter() - started)
            np.testing.assert_array_equal(result, reference)

        expected = float(reference.sum())
        np.testing.assert_allclose(checksums, expected, rtol=1e-6, atol=1e-6)
        warm_median = statistics.median(warm)

        print(f"Platform: {sys.platform}")
        print(f"Architecture: {__import__('platform').machine()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"JAX: {jax.__version__}")
        print(f"Equinox: {eqx.__version__}")
        print("Model: MLP 10 -> 128 -> 128 -> 128 -> 1")
        print(f"Batch rows: {ROWS}")
        print(f"FNOM size: {manifest.stat().st_size / 1000:.3f} kB")
        print(f"Cold activation: {series(cold)}")
        print(f"Warm inference: {series(warm)}")
        print(f"Warm throughput: {ROWS / warm_median:.0f} rows/s")
        print("Parity: PASS")
        print("FoamNordic Equinox runtime benchmark: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    arguments = parser.parse_args()
    if arguments.child is not None:
        child(arguments.child)
    else:
        benchmark()


if __name__ == "__main__":
    main()
