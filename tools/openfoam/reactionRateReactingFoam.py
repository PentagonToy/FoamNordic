#!/usr/bin/env python3
"""Run reactionRateFjord inside an unmodified stock reactingFoam executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter

import numpy as np
from sklearn.linear_model import LinearRegression

import foamnordic as fno


SOURCE_DIMENSIONS = {
    "volumetric_mass": "[1 -3 -1 0 0 0 0]",
    "specific": "[0 0 -1 0 0 0 0]",
}
SOURCE_BASIS = {
    "volumetric_mass": "volumetricMass",
    "specific": "specific",
}


def _foam_dictionary(path: Path, entry: str, value: str) -> None:
    subprocess.run(
        ["foamDictionary", str(path), "-entry", entry, "-set", value],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _source_field(
    source: Path, destination: Path, name: str, dimensions: str
) -> None:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(\s*object\s+)\w+(\s*;)", rf"\g<1>{name}\g<2>", text)
    text = re.sub(
        r"(?m)^(\s*dimensions\s+)\[[^]]+\](\s*;)",
        rf"\g<1>{dimensions}\g<2>",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*internalField\s+)uniform\s+[^;]+;",
        r"\g<1>uniform 0;",
        text,
    )
    destination.write_text(text, encoding="utf-8")


def _prepare(
    source: Path, workspace: Path, end_time: float, dimensions: str
) -> Path:
    prepared = workspace / "source"
    shutil.copytree(source, prepared)
    initial = prepared / "0"
    _source_field(
        initial / "CH4", initial / "omega_CH4", "omega_CH4", dimensions
    )
    control = prepared / "system/controlDict"
    for entry, value in (
        ("startFrom", "startTime"),
        ("startTime", "0"),
        ("endTime", f"{end_time:g}"),
        ("writeControl", "timeStep"),
        ("writeInterval", "1"),
    ):
        _foam_dictionary(control, entry, value)
    subprocess.run(
        ["blockMesh", "-case", str(prepared)],
        check=True,
        stdout=(workspace / "mesh.log").open("wb"),
        stderr=subprocess.STDOUT,
    )
    return prepared


def _artifact(workspace: Path) -> Path:
    features = np.array(
        [
            [0.0, 0.0, 600.0],
            [0.1, 0.1, 1000.0],
            [0.3, 0.2, 1500.0],
            [0.6, 0.1, 2000.0],
        ],
        dtype=np.float64,
    )
    # A small positive source is sufficient to prove model dispatch and the
    # standard reactingFoam species-equation boundary without destabilising
    # the reduced chemistry trajectory.
    source = 1.0e-4 * (1.0 - features[:, 0])
    model = LinearRegression().fit(features, source)
    return fno.export.joblib(
        model,
        path=workspace / "models/reaction-rate.fnom",
        inputs={
            "progress": fno.Tensor.scalar(),
            "variance": fno.Tensor.scalar(),
            "temperature": fno.Tensor.scalar(),
        },
        outputs={"reaction_rate": fno.Tensor.scalar()},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--end-time", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--reaction-rate-basis",
        choices=tuple(SOURCE_DIMENSIONS),
        default="volumetric_mass",
    )
    arguments = parser.parse_args()

    workspace = arguments.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    source = _prepare(
        arguments.source.expanduser().resolve(),
        workspace,
        arguments.end_time,
        SOURCE_DIMENSIONS[arguments.reaction_rate_basis],
    )
    artifact = _artifact(workspace)
    repository = Path(__file__).resolve().parents[2]
    integration = fno.OpenFOAM.DictionaryTemplate(
        source=(
            repository
            / "src/foamnordic/template/openfoam/combustion-model"
            / "reactionRateFjordProperties.in"
        ),
        destination="constant/combustionProperties",
        variables={
            "PROGRESS_FIELD": "CH4",
            "REACTION_RATE_DIMENSIONS": SOURCE_DIMENSIONS[
                arguments.reaction_rate_basis
            ],
            "REACTION_RATE_BASIS": SOURCE_BASIS[arguments.reaction_rate_basis],
        },
    )
    case = fno.OpenFOAM.Case(
        name="reactionRateReactingFoam",
        case_dir=source,
        run_dir=workspace / "output",
        of_cmd=None,
        application="reactingFoam",
        ranks=1,
        integration=integration,
    )
    closure = fno.Closure(
        name="reactionRateFjord",
        operator=fno.Operator.model(artifact),
        inputs={
            "progress": fno.field("CH4"),
            "variance": fno.field("O2"),
            "temperature": fno.field("T"),
        },
        outputs={"reaction_rate": fno.field("omega_CH4")},
    )
    started = perf_counter()
    run = fno.Longship(case=case, closures=(closure,)).launch(
        start_timeout=arguments.timeout, verbose=False
    )
    result = run.stop(force=False, timeout=arguments.timeout)
    elapsed = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"reactionRateFjord reactingFoam gate failed: {result}")

    source_values = result.postprocess.field("omega_CH4", time_idx=-1)
    if not np.all(np.isfinite(source_values)) or not np.any(source_values > 0.0):
        raise RuntimeError("reactionRateFjord did not publish a finite positive source")
    report = {
        "schema": "foamnordic.reaction-rate-reacting-foam/v1",
        "application": "reactingFoam",
        "source": str(arguments.source.expanduser().resolve()),
        "cells": int(source_values.size),
        "elapsed_seconds": elapsed,
        "source_field": "omega_CH4",
        "reaction_rate_basis": arguments.reaction_rate_basis,
        "source_dimensions": SOURCE_DIMENSIONS[arguments.reaction_rate_basis],
        "source_minimum": float(source_values.min()),
        "source_maximum": float(source_values.max()),
        "run": str(result.work_dir),
        "passed": True,
    }
    report_path = workspace / "reaction-rate-reacting-foam.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
