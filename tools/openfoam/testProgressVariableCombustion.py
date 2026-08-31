#!/usr/bin/env python3
"""Run a reduced, solver-integrated progress-variable combustion gate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from sklearn.linear_model import LinearRegression

import foamnordic as fno


def _field(
    source: Path,
    destination: Path,
    name: str,
    dimensions: str,
    value: float,
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
        rf"\g<1>uniform {value};",
        text,
    )
    destination.write_text(text, encoding="utf-8")


def _foam_dictionary(path: Path, entry: str, action: str, value: str) -> None:
    subprocess.run(
        ["foamDictionary", str(path), "-entry", entry, action, value],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _prepare_source(
    source: Path,
    root: Path,
    reaction_rate_dimensions: str,
) -> Path:
    case = root / "source"
    shutil.copytree(source, case)
    initial = case / "0"
    scalar = initial / "CH4"
    _field(scalar, initial / "c_tilde", "c_tilde", "[0 0 0 0 0 0 0]", 0.25)
    _field(scalar, initial / "c_var", "c_var", "[0 0 0 0 0 0 0]", 0.01)
    _field(
        scalar,
        initial / "omega_c",
        "omega_c",
        reaction_rate_dimensions,
        0.0,
    )

    control = case / "system/controlDict"
    schemes = case / "system/fvSchemes"
    solution = case / "system/fvSolution"
    for entry, value in (
        ("startFrom", "startTime"),
        ("startTime", "0"),
        ("endTime", "3e-6"),
        ("deltaT", "1e-6"),
        ("writeControl", "timeStep"),
        ("writeInterval", "1"),
    ):
        _foam_dictionary(control, entry, "-set", value)
    # The deliberately narrow reference solver uses a global time step and
    # does not create reactingFoam's LTS rDeltaT field. Normalize copied LTS
    # cases instead of inheriting a solver-specific ddt scheme implicitly.
    _foam_dictionary(schemes, "ddtSchemes/default", "-set", "Euler")
    for field in ("c_tilde", "c_var"):
        _foam_dictionary(
            schemes,
            f"divSchemes/div(phi,{field})",
            "-add",
            "Gauss limitedLinear 1",
        )
        _foam_dictionary(
            solution,
            f"solvers/{field}",
            "-add",
            "{ solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0; }",
        )
        _foam_dictionary(
            solution,
            f"solvers/{field}Final",
            "-add",
            "{ solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0; }",
        )
    subprocess.run(["blockMesh", "-case", str(case)], check=True)
    return case


def _artifacts(root: Path) -> tuple[Path, Path]:
    model_dir = root / "models"
    model_dir.mkdir()
    features = np.array(
        [
            [0.0, 0.00, 300.0],
            [0.3, 0.01, 800.0],
            [0.7, 0.02, 1200.0],
            [1.0, 0.00, 1800.0],
        ],
        dtype=np.float64,
    )
    source = 2.0e-2 * (1.0 - features[:, 0])
    reaction_model = LinearRegression().fit(features, source)
    reaction = fno.export.joblib(
        reaction_model,
        path=model_dir / "reaction-rate.fnom",
        inputs={
            "progress": fno.Tensor.scalar(),
            "variance": fno.Tensor.scalar(),
            "temperature": fno.Tensor.scalar(),
        },
        outputs={"reaction_rate": fno.Tensor.scalar()},
    )

    manifold_features = features[:, :2]
    species = np.column_stack(
        (1.0 - manifold_features[:, 0], manifold_features[:, 0])
    )
    manifold_model = LinearRegression().fit(manifold_features, species)
    manifold = fno.export.joblib(
        manifold_model,
        path=model_dir / "flamelet.fnom",
        inputs={
            "progress": fno.Tensor.scalar(),
            "variance": fno.Tensor.scalar(),
        },
        outputs={
            "CH4": fno.Tensor.scalar(),
            "CO2": fno.Tensor.scalar(),
        },
    )
    return reaction, manifold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--reaction-rate-basis",
        choices=("volumetric_mass", "specific"),
        default="volumetric_mass",
    )
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("--ranks must be positive")

    root = args.workspace.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    reaction_rate_dimensions = (
        "[1 -3 -1 0 0 0 0]"
        if args.reaction_rate_basis == "volumetric_mass"
        else "[0 0 -1 0 0 0 0]"
    )
    case_dir = _prepare_source(
        args.source.expanduser().resolve(), root, reaction_rate_dimensions
    )
    reaction_artifact, manifold_artifact = _artifacts(root)
    runtime = args.runtime.expanduser().resolve()
    os.environ["FOAMNORDIC_OPENFOAM_LIB"] = str(runtime / "lib")

    case = fno.OpenFOAM.Case(
        name="progressVariableCombustionGate",
        case_dir=case_dir,
        run_dir=root / "output",
        of_cmd=None,
        application="foamnordicProgressVariableFoam",
        ranks=args.ranks,
    )
    reaction_rate = fno.Closure(
        name="reactionRate",
        operator=fno.Operator.model(reaction_artifact),
        inputs={
            "progress": fno.field("c_tilde"),
            "variance": fno.field("c_var"),
            "temperature": fno.field("T"),
        },
        outputs={"reaction_rate": fno.field("omega_c")},
    )
    table = fno.Combustion.Manifold.beta_fdf(
        table=manifold_artifact,
        progress=fno.field("c_tilde"),
        variance=fno.field("c_var"),
        outputs={"species": fno.fields("C*")},
    )
    declaration = fno.Combustion.ProgressVariable(
        reaction_rate=reaction_rate,
        manifold=table,
        coupling=fno.Combustion.CouplingPolicy(
            reaction_rate_basis=args.reaction_rate_basis
        ),
    )
    run = fno.Longship(case=case, combustion=declaration).launch(
        start_timeout=120
    )
    result = run.stop(force=False, timeout=180)
    summary = result.summary(style="compact")
    if summary.status != "succeeded":
        raise RuntimeError(f"solver-integrated combustion gate failed: {summary}")
    final = result.postprocess
    progress = final.field("c_tilde", time_idx=-1)
    source = final.field("omega_c", time_idx=-1)
    methane = final.field("CH4", time_idx=-1)
    carbon_dioxide = final.field("CO2", time_idx=-1)
    print(
        "[FoamNordic] Progress-variable solver gate: PASS\n"
        f"[FoamNordic] Ranks: {args.ranks}\n"
        f"[FoamNordic] Reaction-rate basis: {args.reaction_rate_basis}\n"
        f"[FoamNordic] c range: {progress.min():.9e} .. {progress.max():.9e}\n"
        f"[FoamNordic] omega range: {source.min():.9e} .. {source.max():.9e}\n"
        f"[FoamNordic] CH4+CO2 max error: "
        f"{np.max(np.abs(methane + carbon_dioxide - 1.0)):.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
