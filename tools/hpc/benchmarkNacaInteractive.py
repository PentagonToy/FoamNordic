#!/usr/bin/env python3
"""Benchmark OpenFOAM via FoamNordic and native kEqnFjord transport."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

import foamnordic as fno


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
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--end-time", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--relative-l2-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--peak-relative-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    return parser.parse_args()


def replace_entry(text: str, name: str, value: str) -> str:
    pattern = rf"(?m)^(\s*{re.escape(name)}\s+)[^;]+;"
    rendered, count = re.subn(pattern, rf"\g<1>{value};", text, count=1)
    if count != 1:
        raise RuntimeError(f"controlDict entry not found: {name}")
    return rendered


def stage_source(source: Path, output: Path, end_time: float) -> Path:
    destination = output / "source"
    shutil.copytree(source, destination)
    for child in tuple(destination.iterdir()):
        if child.name.startswith("processor") and child.is_dir():
            shutil.rmtree(child)
            continue
        try:
            numeric = float(child.name)
        except ValueError:
            continue
        if child.is_dir() and numeric != 0.0:
            shutil.rmtree(child)
    shutil.rmtree(destination / "postProcessing", ignore_errors=True)
    control = destination / "system/controlDict"
    rendered = control.read_text(encoding="utf-8")
    for name, value in (
        ("startFrom", "startTime"),
        ("startTime", "0"),
        ("endTime", f"{end_time:.16g}"),
        ("writeControl", "runTime"),
        ("writeInterval", f"{end_time:.16g}"),
        ("writeFormat", "ascii"),
        ("writePrecision", "16"),
    ):
        rendered = replace_entry(rendered, name, value)
    control.write_text(rendered, encoding="utf-8")
    return destination


def closure() -> fno.Closure:
    declared = fno.Closure(
        name="kEqnFjord",
        operator=fno.Operator.function(keqn_function),
        inputs={
            "k": fno.Field("k"),
            "velocity_grad": fno.Field.grad("U"),
            "nu": fno.Field("nu"),
            "delta": fno.Field.delta(),
        },
        outputs={
            "k": fno.Field("k"),
            "nut": fno.Field("nut"),
        },
    )
    if declared.execution_lane != "native":
        raise RuntimeError("kEqnFjord benchmark function did not lower natively")
    return declared


def case(name: str, source: Path, output: Path, ranks: int) -> fno.OpenFOAM.Case:
    value = fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd="openfoam/2512",
        shell="bash",
        application="pimpleFoam",
        ranks=ranks,
    )
    return value.initialize(ranks=ranks, mesh="blockMesh", validate_mesh=True)


def solver_timing(path: Path) -> tuple[float, float]:
    matches = re.findall(
        r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s+s\s+"
        r"ClockTime\s*=\s*([0-9.eE+-]+)\s+s",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        raise RuntimeError(f"OpenFOAM timing not found: {path}")
    execution, clock = matches[-1]
    return float(execution), float(clock)


def launch(
    label: str,
    source: Path,
    output: Path,
    ranks: int,
    timeout: float,
    coupled: bool,
) -> tuple[fno.Runtime.Result, dict[str, object]]:
    declared = fno.Longship(
        case=case(label, source, output, ranks),
        closures=(closure(),) if coupled else (),
    )
    started = perf_counter()
    result = declared.launch(start_timeout=timeout, verbose=False).stop(
        timeout=timeout
    )
    wall = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"{label} failed; see {result.solver_log}")
    execution, clock = solver_timing(result.solver_log)
    return result, {
        "label": label,
        "mode": "fno-native" if coupled else "openfoam-via-foamnordic",
        "wall_seconds": wall,
        "result_elapsed_seconds": result.elapsed_seconds,
        "openfoam_execution_seconds": execution,
        "openfoam_clock_seconds": clock,
        "run": str(result.work_dir),
    }


def compare(
    via_foamnordic: fno.Runtime.Result,
    coupled: fno.Runtime.Result,
    relative_l2_tolerance: float,
    peak_relative_tolerance: float,
    atol: float,
) -> dict[str, dict[str, float | bool]]:
    metrics = fno.Postprocess.compare(
        via_foamnordic,
        coupled,
        fields=("U", "p", "nut", "k"),
        time_idx=-1,
        mesh="strict",
        verbose=False,
    )
    passed = True
    for field, values in metrics.items():
        reference = np.asarray(
            via_foamnordic.postprocess.field(field, time_idx=-1)
        )
        scale = float(np.max(np.abs(reference)))
        peak_relative = values["max_abs"] / max(scale, atol)
        field_passed = bool(
            peak_relative <= peak_relative_tolerance
            and values["relative_l2"] <= relative_l2_tolerance
        )
        values["reference_max_abs"] = scale
        values["peak_relative"] = peak_relative
        values["passed"] = field_passed
        passed = passed and field_passed
    print("[FoamNordic] Integrated field comparison:")
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise RuntimeError(
            "OpenFOAM-via-FoamNordic and native kEqnFjord fields exceeded "
            "the declared tolerances"
        )
    return metrics


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
    root = options.output.expanduser().resolve() / f"naca-keqn-{stamp}"
    root.mkdir(parents=True)
    source = stage_source(options.source.expanduser().resolve(), root, options.end_time)

    print("[FoamNordic] Warm-up: OpenFOAM via FoamNordic")
    launch("warmup-openfoam-via-foamnordic", source, root, options.ranks, options.timeout, False)
    print("[FoamNordic] Warm-up: FNO-native")
    launch("warmup-fno-native", source, root, options.ranks, options.timeout, True)

    records = []
    comparisons = []
    for repetition in range(1, options.repetitions + 1):
        order = (False, True) if repetition % 2 else (True, False)
        pair = {}
        for coupled in order:
            mode = "fno-native" if coupled else "openfoam-via-foamnordic"
            label = f"naca-{repetition:02d}-{mode}"
            print(f"[FoamNordic] Measured run {repetition}: {mode}")
            result, record = launch(
                label, source, root, options.ranks, options.timeout, coupled
            )
            records.append(record)
            pair[mode] = result
        comparisons.append(
            compare(
                pair["openfoam-via-foamnordic"],
                pair["fno-native"],
                options.relative_l2_tolerance,
                options.peak_relative_tolerance,
                options.atol,
            )
        )

    via_foamnordic = [
        record for record in records
        if record["mode"] == "openfoam-via-foamnordic"
    ]
    native = [
        record for record in records if record["mode"] == "fno-native"
    ]
    report = {
        "schema": "foamnordic.naca-keqn-benchmark",
        "source": str(options.source.expanduser().resolve()),
        "ranks": options.ranks,
        "end_time": options.end_time,
        "repetitions": options.repetitions,
        "acceptance": {
            "relative_l2_tolerance": options.relative_l2_tolerance,
            "peak_relative_tolerance": options.peak_relative_tolerance,
            "absolute_floor": options.atol,
        },
        "openfoam_via_foamnordic": {
            "wall_seconds": reduced(via_foamnordic, "wall_seconds"),
            "openfoam_execution_seconds": reduced(
                via_foamnordic, "openfoam_execution_seconds"
            ),
            "openfoam_clock_seconds": reduced(
                via_foamnordic, "openfoam_clock_seconds"
            ),
        },
        "fno_native": {
            "wall_seconds": reduced(native, "wall_seconds"),
            "openfoam_execution_seconds": reduced(
                native, "openfoam_execution_seconds"
            ),
            "openfoam_clock_seconds": reduced(
                native, "openfoam_clock_seconds"
            ),
        },
        "records": records,
        "comparisons": comparisons,
        "passed": True,
    }
    report["wall_overhead_ratio"] = (
        report["fno_native"]["wall_seconds"]["median"]
        / report["openfoam_via_foamnordic"]["wall_seconds"]["median"]
    )
    path = root / "benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FoamNordic] Report: {path}")
    print("[FoamNordic] NACA4412 native kEqn benchmark: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
