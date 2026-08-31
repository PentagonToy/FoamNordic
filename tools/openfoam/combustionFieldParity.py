#!/usr/bin/env python3
"""Validate a reactingFoam case and an identity thermochemical field exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from time import perf_counter

import numpy as np

import foamnordic as fno


DEFAULT_FIELDS = ("U", "p", "T", "CH4", "O2", "CO2", "H2O", "Qdot")


def identity_thermochemistry(temperature, fuel):
    """Return two solver-owned fields unchanged through the resident backend."""

    return {"temperature": temperature, "fuel": fuel}


def _run_openfoam(command: str, *, output: Path | None = None) -> None:
    if output is None:
        subprocess.run(
            ["openfoam", "-c", command],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["openfoam", "-c", command],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _foam_set(path: Path, entry: str, value: str) -> None:
    command = (
        f"foamDictionary {shlex.quote(str(path))} "
        f"-entry {shlex.quote(entry)} -set {shlex.quote(value)}"
    )
    _run_openfoam(command)


def _prepare_source(
    source: Path,
    destination: Path,
    end_time: float,
) -> tuple[int | None, Path]:
    shutil.copytree(source, destination)
    for child in tuple(destination.iterdir()):
        try:
            numeric = float(child.name)
        except ValueError:
            numeric = None
        if child.is_dir() and numeric is not None and numeric != 0.0:
            shutil.rmtree(child)
        elif child.is_dir() and (
            child.name.startswith("processor")
            or child.name in {"postProcessing", "TDAC"}
        ):
            shutil.rmtree(child)

    control = destination / "system/controlDict"
    _foam_set(control, "startFrom", "startTime")
    _foam_set(control, "startTime", "0")
    _foam_set(control, "endTime", f"{end_time:.16g}")
    _foam_set(control, "writeControl", "runTime")
    _foam_set(control, "writeInterval", f"{end_time:.16g}")
    _foam_set(control, "writePrecision", "16")

    mesh_log = destination.parent / "mesh.log"
    mesh = destination / "constant/polyMesh/points"
    block_dictionary = destination / "system/blockMeshDict"
    if not mesh.is_file():
        if not block_dictionary.is_file():
            raise FileNotFoundError(
                "combustion source has neither an existing mesh nor blockMeshDict"
            )
        command = (
            f"blockMesh -case {shlex.quote(str(destination))} && "
            f"checkMesh -case {shlex.quote(str(destination))}"
        )
    else:
        command = f"checkMesh -case {shlex.quote(str(destination))}"
    _run_openfoam(command, output=mesh_log)
    match = re.search(r"\bcells:\s*(\d+)", mesh_log.read_text(encoding="utf-8"))
    return (None if match is None else int(match.group(1))), mesh_log


def _case(name: str, source: Path, output: Path, application: str) -> fno.OpenFOAM.Case:
    return fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd="openfoam",
        shell="zsh",
        application=application,
        ranks=1,
    )


def _launch(longship: fno.Longship, timeout: float):
    started = perf_counter()
    run = longship.launch(start_timeout=timeout, verbose=False)
    result = run.stop(timeout=timeout)
    wall = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"{longship.name} failed; see {result.solver_log}")
    return run, result, wall


def _solver_facts(path: Path) -> dict[str, int | float | None]:
    contents = path.read_text(encoding="utf-8", errors="replace")
    timing = re.findall(
        r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s+s\s+"
        r"ClockTime\s*=\s*([0-9.eE+-]+)\s+s",
        contents,
    )
    chemistry = re.search(
        r"Number of species\s*=\s*(\d+)\s+and reactions\s*=\s*(\d+)",
        contents,
    )
    return {
        "execution_seconds": float(timing[-1][0]) if timing else None,
        "clock_seconds": float(timing[-1][1]) if timing else None,
        "species": int(chemistry.group(1)) if chemistry else None,
        "reactions": int(chemistry.group(2)) if chemistry else None,
    }


def _observations(run) -> list[dict[str, object]]:
    records = []
    for record in run.observe():
        records.append(
            {
                "exchange_index": record.exchange_index,
                "physical_time": record.time,
                "summary": {
                    name: {
                        "minimum": field.minimum,
                        "maximum": field.maximum,
                        "mean": field.mean,
                        "l2": field.l2,
                        "count": field.count,
                    }
                    for name, field in record.summary.items()
                },
                "timing": {
                    "closure_wait": record.timing.closure_wait,
                    "evaluate": record.timing.evaluate,
                },
            }
        )
    return records


def _comparison(
    stock: fno.Result,
    coupled: fno.Result,
    fields: tuple[str, ...],
    rtol: float,
    atol: float,
) -> tuple[dict[str, dict[str, object]], bool]:
    metrics = fno.Postprocess.compare(
        stock,
        coupled,
        fields=fields,
        time_idx=-1,
        mesh="strict",
        verbose=False,
    )
    passed = True
    for field, values in metrics.items():
        reference = np.asarray(
            stock.postprocess.field(field, time_idx=-1), dtype=np.float64
        )
        scale = float(np.max(np.abs(reference)))
        values["reference_max_abs"] = scale
        values["relative_linf"] = values["max_abs"] / max(
            scale, np.finfo(float).tiny
        )
        field_passed = (
            values["max_abs"] <= atol + rtol * scale
            and values["relative_l2"] <= rtol
        )
        values["passed"] = field_passed
        passed = passed and field_passed
    return metrics, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--application", default="reactingFoam")
    parser.add_argument("--temperature-field", default="T")
    parser.add_argument("--fuel-field", default="CH4")
    parser.add_argument("--end-time", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="comma-separated final fields compared against stock OpenFOAM",
    )
    arguments = parser.parse_args()

    source = arguments.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source case does not exist: {source}")
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else Path(tempfile.mkdtemp(prefix="foamnordic-combustion-parity."))
    )
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "source"
    cells, mesh_log = _prepare_source(source, prepared, arguments.end_time)

    discovered = _case(
        "combustion-fields", prepared, output, arguments.application
    ).fields
    required = (arguments.temperature_field, arguments.fuel_field)
    missing = tuple(field for field in required if field not in discovered)
    if missing:
        parser.error(
            f"identity exchange fields are missing: {', '.join(missing)}; "
            f"available: {', '.join(discovered)}"
        )
    requested = tuple(
        field.strip() for field in arguments.fields.split(",") if field.strip()
    )
    _, stock, stock_wall = _launch(
        fno.Longship(
            case=_case(
                "combustion-stock", prepared, output, arguments.application
            )
        ),
        arguments.timeout,
    )
    transform = fno.Transform(
        name="identityThermochemistry",
        operator=fno.Operator.function(identity_thermochemistry),
        inputs={
            "temperature": fno.field(arguments.temperature_field),
            "fuel": fno.field(arguments.fuel_field),
        },
        outputs={
            "temperature": fno.field(arguments.temperature_field),
            "fuel": fno.field(arguments.fuel_field),
        },
        at="time_step_start",
    )
    run, coupled, coupled_wall = _launch(
        fno.Longship(
            case=_case(
                "combustion-identity", prepared, output, arguments.application
            ),
            transforms=(transform,),
            observations=(
                fno.Observe(
                    summaries={
                        arguments.temperature_field: ("min", "max", "mean"),
                        arguments.fuel_field: ("min", "max", "mean"),
                    },
                    interval=1,
                ),
            ),
        ),
        arguments.timeout,
    )
    comparison, passed = _comparison(
        stock,
        coupled,
        requested,
        arguments.rtol,
        arguments.atol,
    )
    report = {
        "schema": "foamnordic.combustion-field-parity/v1",
        "source": str(source),
        "application": arguments.application,
        "cells": cells,
        "end_time": arguments.end_time,
        "exchange_fields": list(required),
        "comparison_fields": list(requested),
        "rtol": arguments.rtol,
        "atol": arguments.atol,
        "stock": {
            "total_seconds": stock_wall,
            **_solver_facts(stock.solver_log),
            "run": str(stock.work_dir),
        },
        "foamnordic": {
            "total_seconds": coupled_wall,
            **_solver_facts(coupled.solver_log),
            "run": str(coupled.work_dir),
        },
        "observations": _observations(run),
        "fields": comparison,
        "mesh_log": str(mesh_log),
        "passed": passed,
    }
    report_path = output / "combustion-field-parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
