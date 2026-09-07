#!/usr/bin/env python3
"""Benchmark OF/FNO WALE against FNO-coupled nutFjord."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import foamnordic as fno

import benchmarkNacaInteractive as benchmark
import validateNutClosure as nutfjord


def wale_function(velocity_grad, delta, C_w=0.325):
    squared = fno.Math.matmul(velocity_grad, velocity_grad)
    traceless = fno.Math.dev(fno.Math.symm(squared))
    strain = fno.Math.symm(velocity_grad)
    sd2 = fno.Math.maximum(fno.Math.ddot(traceless, traceless), 0.0)
    s2 = fno.Math.maximum(fno.Math.ddot(strain, strain), 0.0)
    numerator = sd2**1.5
    denominator = s2**2.5 + sd2**1.25
    safe = fno.Math.where(denominator > 0.0, denominator, 1.0)
    nut = (C_w * delta) ** 2 * numerator / safe
    return fno.Math.where(denominator > 0.0, nut, 0.0)


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
    root = options.output.expanduser().resolve() / f"pitzdaily-wale-{stamp}"
    root.mkdir(parents=True)
    source = benchmark.stage_source(
        options.source.expanduser().resolve(), root, options.end_time
    )
    nutfjord.deterministic_decomposition(source, options.ranks)

    modes = (
        ("openfoam-via-foamnordic", None),
        ("fno-coupled", wale_function),
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
                f"wale-{repetition:02d}-{mode}", source, root,
                options.ranks, options.timeout, function,
            )
            timing["mode"] = mode
            timing["repetition"] = repetition
            timing["run"] = str(result.work_dir)
            records.append(timing)
            pair[mode] = result
        comparisons.append(
            nutfjord.compare(
                pair["openfoam-via-foamnordic"], pair["fno-coupled"]
            )
        )

    report: dict[str, object] = {
        "schema": "foamnordic.pitzdaily-wale-benchmark",
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
    coupled = report["fno_coupled"]
    report["wall_ratio"] = (
        coupled["wall_seconds"]["median"]
        / reference["wall_seconds"]["median"]
    )
    report["openfoam_execution_ratio"] = (
        coupled["openfoam_execution_seconds"]["median"]
        / reference["openfoam_execution_seconds"]["median"]
    )
    path = root / "benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FoamNordic] Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
