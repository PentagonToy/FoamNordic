#!/usr/bin/env python3
"""Benchmark OF/FNO Smagorinsky against FNO-native nutFjord."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import foamnordic as fno

import benchmarkNacaInteractive as benchmark
import validateNutClosure as nutfjord


def smagorinsky_function(velocity_grad, delta, C_k=0.0265463553, C_e=1.048):
    strain = fno.Math.symm(velocity_grad)
    trace = fno.Math.einsum("...ii->...", strain)
    deviatoric = fno.Math.dev(strain)
    a = C_e / delta
    b = (2.0 / 3.0) * trace
    c = 2.0 * C_k * delta * fno.Math.ddot(deviatoric, strain)
    discriminant = b**2 + 4.0 * a * c
    sqrt_k = (-b + fno.Math.sqrt(fno.Math.maximum(discriminant, 0.0))) / (2.0 * a)
    return C_k * delta * fno.Math.maximum(sqrt_k, 0.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--end-time", type=float, default=4.0e-3)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def reduced(records: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records]
    median = statistics.median(values)
    return {
        "median": median,
        "maximum_deviation": max(abs(value - median) for value in values),
    }


def main() -> int:
    options = arguments()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = options.output.expanduser().resolve() / f"pitzdaily-smagorinsky-{stamp}"
    root.mkdir(parents=True)
    source = benchmark.stage_source(
        options.source.expanduser().resolve(), root, options.end_time
    )
    nutfjord.deterministic_decomposition(source, options.ranks)

    modes = (
        ("openfoam-via-foamnordic", None),
        ("fno-native", smagorinsky_function),
    )
    for mode, function in modes:
        print(f"[FoamNordic] Warm-up: {mode}")
        nutfjord.launch(
            f"warmup-{mode}", source, root, options.ranks, options.timeout,
            function,
        )

    records: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for repetition in range(1, options.repetitions + 1):
        ordered = modes if repetition % 2 else tuple(reversed(modes))
        pair = {}
        for mode, function in ordered:
            print(f"[FoamNordic] Measured run {repetition}: {mode}")
            result, timing = nutfjord.launch(
                f"smagorinsky-{repetition:02d}-{mode}", source, root,
                options.ranks, options.timeout, function,
            )
            timing["mode"] = mode
            timing["repetition"] = repetition
            timing["run"] = str(result.work_dir)
            records.append(timing)
            pair[mode] = result
        comparisons.append(
            nutfjord.compare(
                pair["openfoam-via-foamnordic"], pair["fno-native"]
            )
        )

    report: dict[str, object] = {
        "schema": "foamnordic.pitzdaily-smagorinsky-benchmark",
        "source": str(options.source.expanduser().resolve()),
        "ranks": options.ranks,
        "end_time": options.end_time,
        "repetitions": options.repetitions,
        "records": records,
        "comparisons": comparisons,
    }
    for mode, _ in modes:
        selected = [record for record in records if record["mode"] == mode]
        report[mode.replace("-", "_")] = {
            key: reduced(selected, key)
            for key in (
                "wall_seconds",
                "openfoam_execution_seconds",
                "openfoam_clock_seconds",
            )
        }
    reference = report["openfoam_via_foamnordic"]
    native = report["fno_native"]
    report["wall_ratio"] = (
        native["wall_seconds"]["median"] / reference["wall_seconds"]["median"]
    )
    report["openfoam_execution_ratio"] = (
        native["openfoam_execution_seconds"]["median"]
        / reference["openfoam_execution_seconds"]["median"]
    )
    path = root / "benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FoamNordic] Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
