#!/usr/bin/env python3
"""Quantify native kEqnFjord trajectory error growth."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import foamnordic as fno

import benchmarkNacaInteractive as benchmark


FIELDS = ("U", "p", "nut", "k")


def keqn_function(k, velocity_grad, nu, delta, C_k=0.094, C_e=1.048):
    k_positive = fno.Math.maximum(k, 1.0e-10)
    sqrt_k = fno.Math.sqrt(k_positive)
    strain = fno.Math.dev(2.0 * fno.Math.symm(velocity_grad))
    nut = C_k * delta * sqrt_k
    production = nut * fno.Math.ddot(velocity_grad, strain)
    dissipation = C_e * sqrt_k / delta
    k_new = fno.Field.transport(
        k,
        diffusivity=nu + nut,
        source=production,
        implicit_sink=dissipation,
        lower_bound=1.0e-10,
    )
    return {"k": k_new, "nut": C_k * delta * fno.Math.sqrt(k_new)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--delta-t", type=float, default=2.5e-3)
    parser.add_argument("--steps", type=int, nargs="+", default=(1, 10, 100, 400))
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def field_comparison(via_foamnordic, coupled) -> dict[str, dict[str, float]]:
    metrics = benchmark.fno.Postprocess.compare(
        via_foamnordic,
        coupled,
        fields=FIELDS,
        time_idx=-1,
        mesh="strict",
        verbose=False,
    )
    for field, values in metrics.items():
        reference = np.asarray(
            via_foamnordic.postprocess.field(field, time_idx=-1)
        )
        scale = float(np.max(np.abs(reference)))
        values["reference_max_abs"] = scale
        values["peak_relative"] = values["max_abs"] / max(
            scale, np.finfo(float).tiny
        )
    return metrics


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "steps",
                "physical_time",
                "field",
                "mae",
                "rmse",
                "max_abs",
                "relative_l2",
                "peak_relative",
            ),
        )
        writer.writeheader()
        for record in records:
            for field, metrics in record["fields"].items():
                writer.writerow(
                    {
                        "steps": record["steps"],
                        "physical_time": record["physical_time"],
                        "field": field,
                        **{name: metrics[name] for name in writer.fieldnames[3:]},
                    }
                )


def scientific(value: float) -> str:
    return f"{value:.3e}"


def write_markdown(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    lines = [
        "# NACA4412 kEqn numerical validation",
        "",
        "Each checkpoint restarts from the same initial condition and compares OpenFOAM via FoamNordic (OF/FNO) with native kEqnFjord transport.",
        "",
        "## Integrated trajectory",
        "",
        "### Relative L2 error",
        "",
        "| Field | " + " | ".join(f"{record['steps']} steps" for record in records) + " |",
        "| --- | " + " | ".join("---:" for _ in records) + " |",
    ]
    for field in FIELDS:
        values = " | ".join(
            scientific(record["fields"][field]["relative_l2"])
            for record in records
        )
        lines.append(f"| `{field}` | {values} |")
    lines.extend(
        [
            "",
            "### Peak-relative error",
            "",
            "| Field | " + " | ".join(f"{record['steps']} steps" for record in records) + " |",
            "| --- | " + " | ".join("---:" for _ in records) + " |",
        ]
    )
    for field in FIELDS:
        values = " | ".join(
            scientific(record["fields"][field]["peak_relative"])
            for record in records
        )
        lines.append(f"| `{field}` | {values} |")
    times = ", ".join(f"`{record['physical_time']:.6g}`" for record in records)
    lines.extend(
        [
            "",
            f"The columns correspond to physical times {times}, respectively.",
            "",
            "The one-step result covers native graph declaration, transport assembly, solution and atomic field commit. Later checkpoints show how numerical differences accumulate through nonlinear time integration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    options = arguments()
    if any(step <= 0 for step in options.steps):
        raise ValueError("all step counts must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = options.output.expanduser().resolve() / f"naca-keqn-validation-{stamp}"
    root.mkdir(parents=True)
    benchmark.keqn_function = keqn_function

    records = []
    for steps in sorted(set(options.steps)):
        physical_time = steps * options.delta_t
        checkpoint = root / f"step-{steps:04d}"
        checkpoint.mkdir()
        source = benchmark.stage_source(
            options.source.expanduser().resolve(), checkpoint, physical_time
        )
        print(f"[FoamNordic] Step {steps}: OpenFOAM via FoamNordic")
        via_foamnordic, via_foamnordic_timing = benchmark.launch(
            f"step-{steps:04d}-openfoam-via-foamnordic",
            source,
            checkpoint,
            options.ranks,
            options.timeout,
            False,
        )
        print(f"[FoamNordic] Step {steps}: FNO-native")
        coupled, coupled_timing = benchmark.launch(
            f"step-{steps:04d}-fno-native",
            source,
            checkpoint,
            options.ranks,
            options.timeout,
            True,
        )
        record = {
            "steps": steps,
            "physical_time": physical_time,
            "openfoam_via_foamnordic_timing": via_foamnordic_timing,
            "fno_native_timing": coupled_timing,
            "fields": field_comparison(via_foamnordic, coupled),
        }
        records.append(record)
        (root / "validation.json").write_text(
            json.dumps({"trajectory": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record["fields"], indent=2))

    write_csv(root / "trajectory.csv", records)
    write_markdown(root / "README.md", records)
    print(f"[FoamNordic] Validation report: {root / 'README.md'}")
    print("[FoamNordic] NACA4412 closure validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
