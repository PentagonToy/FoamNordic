#!/usr/bin/env python3
"""Validate Smagorinsky and WALE through the nutFjord adapter."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import foamnordic as fno

import benchmarkNacaInteractive as benchmark


FIELDS = ("U", "p", "nut")


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
    parser.add_argument("--smagorinsky-source", type=Path, required=True)
    parser.add_argument("--wale-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--delta-t", type=float, default=1.0e-5)
    parser.add_argument("--steps", type=int, nargs="+", default=(1, 10, 100, 400))
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def deterministic_decomposition(source: Path, ranks: int) -> None:
    """Install a repeatable decomposition so strict cell comparison is valid."""

    path = source / "system/decomposeParDict"
    path.write_text(
        "FoamFile\n"
        "{\n"
        "    format ascii;\n"
        "    class dictionary;\n"
        "    object decomposeParDict;\n"
        "}\n\n"
        f"numberOfSubdomains {ranks};\n"
        "method simple;\n\n"
        "simpleCoeffs\n"
        "{\n"
        f"    n ({ranks} 1 1);\n"
        "    delta 0.001;\n"
        "}\n",
        encoding="utf-8",
    )


def closure(function) -> fno.Closure:
    return fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(function),
        inputs={
            "velocity_grad": fno.Field.grad("U"),
            "delta": fno.Field.delta(),
        },
        outputs={"nut": fno.Field("nut")},
    )


def launch(label, source, output, ranks, timeout, function=None):
    longship = fno.Longship(
        case=benchmark.case(label, source, output, ranks),
        closures=() if function is None else (closure(function),),
    )
    started = perf_counter()
    result = longship.launch(start_timeout=timeout, verbose=False).stop(timeout=timeout)
    wall = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"{label} failed; see {result.solver_log}")
    execution, clock = benchmark.solver_timing(result.solver_log)
    return result, {"wall_seconds": wall, "openfoam_execution_seconds": execution, "openfoam_clock_seconds": clock}


def compare(reference, candidate):
    metrics = fno.Postprocess.compare(reference, candidate, fields=FIELDS, time_idx=-1, mesh="strict", verbose=False)
    for field, values in metrics.items():
        array = np.asarray(reference.postprocess.field(field, time_idx=-1))
        scale = float(np.max(np.abs(array)))
        values["reference_max_abs"] = scale
        values["peak_relative"] = values["max_abs"] / max(scale, np.finfo(float).tiny)
    return metrics


def algebra(function) -> dict[str, object]:
    rng = np.random.default_rng(42)
    gradient = rng.normal(0.0, 2.0, (4096, 3, 3)).astype(np.float64)
    delta = rng.uniform(1.0e-4, 2.0e-2, 4096).astype(np.float64)
    first = np.asarray(function(gradient, delta), dtype=np.float64)
    second = np.asarray(function(gradient.copy(), delta.copy()), dtype=np.float64)
    difference = second - first
    return {
        "dtype": "float64",
        "rows": 4096,
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(first), np.finfo(float).tiny)),
        "finite": bool(np.all(np.isfinite(first))),
        "nonnegative": bool(np.all(first >= 0.0)),
    }


def write_outputs(root: Path, report: dict[str, object]) -> None:
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (root / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("model", "steps", "physical_time", "field", "relative_l2", "peak_relative")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, record in report["models"].items():
            for point in record["trajectory"]:
                for field, values in point["fields"].items():
                    writer.writerow({"model": model, "steps": point["steps"], "physical_time": point["physical_time"], "field": field, "relative_l2": values["relative_l2"], "peak_relative": values["peak_relative"]})
    lines = [
        "# nutFjord numerical validation",
        "",
        "OpenFOAM via FoamNordic (OF/FNO) uses the standard OpenFOAM LES model. FNO-coupled replaces its eddy-viscosity evaluation with the corresponding float64 function through nutFjord.",
    ]
    for model, record in report["models"].items():
        trajectory = record["trajectory"]
        headings = " | ".join(f"{point['steps']} step" if point["steps"] == 1 else f"{point['steps']} steps" for point in trajectory)
        alignment = " | ".join("---:" for _ in trajectory)
        lines.extend(["", f"## {model}", ""])
        for title, metric in (("Relative L2 error", "relative_l2"), ("Peak-relative error", "peak_relative")):
            lines.extend([
                f"### {title}",
                "",
                f"| Field | {headings} |",
                f"| --- | {alignment} |",
            ])
            for field in FIELDS:
                values = " | ".join(f"{point['fields'][field][metric]:.3e}" for point in trajectory)
                lines.append(f"| `{field}` | {values} |")
            lines.append("")
        times = ", ".join(f"`{point['physical_time']:.6g}`" for point in trajectory)
        lines.append(f"The columns correspond to physical times {times}, respectively.")
    lines.append("")
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    options = arguments()
    root = options.output.expanduser().resolve() / f"nutfjord-validation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    root.mkdir(parents=True)
    models = {
        "Smagorinsky": (options.smagorinsky_source.resolve(), smagorinsky_function),
        "WALE": (options.wale_source.resolve(), wale_function),
    }
    report = {"schema": "foamnordic.nutfjord-validation", "ranks": options.ranks, "delta_t": options.delta_t, "models": {}}
    for model, (source_path, function) in models.items():
        print(f"[FoamNordic] {model}: single float64 evaluation")
        record = {"algebra": algebra(function), "trajectory": []}
        report["models"][model] = record
        for steps in sorted(set(options.steps)):
            checkpoint = root / model.lower() / f"step-{steps:04d}"
            checkpoint.mkdir(parents=True)
            source = benchmark.stage_source(source_path, checkpoint, steps * options.delta_t)
            deterministic_decomposition(source, options.ranks)
            print(f"[FoamNordic] {model}, step {steps}: OpenFOAM via FoamNordic")
            reference, reference_timing = launch(f"{model.lower()}-{steps:04d}-openfoam-via-foamnordic", source, checkpoint, options.ranks, options.timeout)
            print(f"[FoamNordic] {model}, step {steps}: FNO-coupled")
            candidate, candidate_timing = launch(f"{model.lower()}-{steps:04d}-fno-coupled", source, checkpoint, options.ranks, options.timeout, function)
            point = {"steps": steps, "physical_time": steps * options.delta_t, "openfoam_via_foamnordic_timing": reference_timing, "fno_coupled_timing": candidate_timing, "fields": compare(reference, candidate)}
            record["trajectory"].append(point)
            write_outputs(root, report)
            print(json.dumps(point["fields"], indent=2))
    print(f"[FoamNordic] Validation report: {root / 'README.md'}")
    print("[FoamNordic] nutFjord validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
