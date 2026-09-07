#!/usr/bin/env python3
"""Validate kEqn algebra and quantify coupled trajectory error growth."""

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


def keqn_function(k, velocity_grad, delta, C_k=0.094, C_e=1.048):
    """Portable closure body serialized from this executable module."""

    k_positive = fno.Math.maximum(k, 0.0)
    sqrt_k = fno.Math.sqrt(k_positive)
    eddy_viscosity = C_k * delta * sqrt_k
    dev_two_symm_grad = fno.Math.dev(
        2.0 * fno.Math.symm(velocity_grad)
    )
    k_production = eddy_viscosity * fno.Math.ddot(
        velocity_grad, dev_two_symm_grad
    )
    k_dissipation_coeff = C_e * sqrt_k / delta
    return eddy_viscosity, k_production, k_dissipation_coeff


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--delta-t", type=float, default=2.5e-3)
    parser.add_argument("--steps", type=int, nargs="+", default=(1, 10, 100, 400))
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def openfoam_algebra(k, velocity_grad, delta, c_k=0.094, c_e=1.048):
    k_positive = np.maximum(k, 0.0)
    sqrt_k = np.sqrt(k_positive)
    nut = c_k * delta * sqrt_k
    symmetric = 0.5 * (velocity_grad + np.swapaxes(velocity_grad, -1, -2))
    two_symmetric = 2.0 * symmetric
    trace = np.trace(two_symmetric, axis1=-2, axis2=-1) / 3.0
    deviatoric = two_symmetric - trace[..., None, None] * np.eye(3)
    production = nut * np.sum(velocity_grad * deviatoric, axis=(-2, -1))
    dissipation = c_e * sqrt_k / delta
    return nut, production, dissipation


def algebra_check() -> dict[str, object]:
    rng = np.random.default_rng(42)
    rows = 4096
    k = rng.uniform(1.0e-12, 2.0e-2, rows).astype(np.float64)
    velocity_grad = rng.normal(0.0, 2.0, (rows, 3, 3)).astype(np.float64)
    delta = rng.uniform(1.0e-4, 2.0e-2, rows).astype(np.float64)
    reference = openfoam_algebra(k, velocity_grad, delta)
    candidate = keqn_function(k, velocity_grad, delta)
    names = ("nut", "kProduction", "kDissipationCoeff")
    outputs = {}
    passed = True
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        difference = np.asarray(actual, dtype=np.float64) - expected
        maximum = float(np.max(np.abs(difference)))
        relative_l2 = float(
            np.linalg.norm(difference) / max(np.linalg.norm(expected), np.finfo(float).tiny)
        )
        output_passed = bool(np.allclose(actual, expected, rtol=1.0e-14, atol=1.0e-15))
        outputs[name] = {
            "max_abs": maximum,
            "relative_l2": relative_l2,
            "passed": output_passed,
        }
        passed = passed and output_passed
    return {"dtype": "float64", "rows": rows, "outputs": outputs, "passed": passed}


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
    algebra: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    lines = [
        "# NACA4412 kEqn numerical validation",
        "",
        "The algebra check compares an independent NumPy transcription of the OpenFOAM kEqn expressions with one eager FoamNordic function evaluation in float64. The trajectory check compares OpenFOAM via FoamNordic (OF/FNO) with FNO-coupled runs from identical initial conditions.",
        "",
        "## Single closure evaluation",
        "",
        "| Output | Maximum absolute error | Relative L2 error | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, values in algebra["outputs"].items():
        lines.append(
            f"| {name} | {scientific(values['max_abs'])} | "
            f"{scientific(values['relative_l2'])} | "
            f"{'PASS' if values['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Integrated trajectory",
            "",
            "### Relative L2 error",
            "",
            "| Field | " + " | ".join(f"{record['steps']} steps" for record in records) + " |",
            "| --- | " + " | ".join("---:" for _ in records) + " |",
        ]
    )
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
            "The one-step result covers packing, Fjord exchange, evaluation, publication, and OpenFOAM field application. Later checkpoints show how small coupled-path rounding differences accumulate through nonlinear time integration.",
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
    # benchmark.closure resolves this global when constructing Operator.function.
    # Keeping the callable in __main__ makes cloudpickle serialize it by value so
    # an independently started ModelHost does not need the tools directory on
    # its import path.
    benchmark.keqn_function = keqn_function
    algebra = algebra_check()
    print("[FoamNordic] Single float64 closure evaluation:")
    print(json.dumps(algebra, indent=2))
    if not algebra["passed"]:
        raise RuntimeError("single closure algebra validation failed")

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
        print(f"[FoamNordic] Step {steps}: FNO-coupled")
        coupled, coupled_timing = benchmark.launch(
            f"step-{steps:04d}-fno-coupled",
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
            "fno_coupled_timing": coupled_timing,
            "fields": field_comparison(via_foamnordic, coupled),
        }
        records.append(record)
        (root / "validation.json").write_text(
            json.dumps({"algebra": algebra, "trajectory": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record["fields"], indent=2))

    write_csv(root / "trajectory.csv", records)
    write_markdown(root / "README.md", algebra, records)
    print(f"[FoamNordic] Validation report: {root / 'README.md'}")
    print("[FoamNordic] NACA4412 closure validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
