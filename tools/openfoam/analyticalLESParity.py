#!/usr/bin/env python3
"""Compare stock OpenFOAM LES closures with exact FoamNordic equivalents."""

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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import foamnordic as fno


jax.config.update("jax_enable_x64", True)

CK = 0.094
CE = 1.048


class ExactSmagorinsky(eqx.Module):
    """OpenFOAM.com Smagorinsky algebra evaluated on one packed cell."""

    ck: jax.Array
    ce: jax.Array

    def __init__(self, ck: float = CK, ce: float = CE):
        self.ck = jnp.asarray([ck], dtype=jnp.float64)
        self.ce = jnp.asarray([ce], dtype=jnp.float64)

    def __call__(self, features):
        gradient = features[:9].reshape(3, 3)
        delta = features[9]
        symmetric = 0.5 * (gradient + gradient.T)
        trace = jnp.trace(symmetric)
        deviatoric = symmetric - (trace / 3.0) * jnp.eye(3, dtype=features.dtype)
        contraction = jnp.sum(deviatoric * symmetric)
        coefficient_a = self.ce[0] / delta
        coefficient_b = (2.0 / 3.0) * trace
        coefficient_c = 2.0 * self.ck[0] * delta * contraction
        root_k = (
            -coefficient_b
            + jnp.sqrt(
                jnp.maximum(
                    coefficient_b**2
                    + 4.0 * coefficient_a * coefficient_c,
                    0.0,
                )
            )
        ) / (2.0 * coefficient_a)
        nut = self.ck[0] * delta * jnp.abs(root_k)
        return jnp.asarray([nut])


class ExactKEqn(eqx.Module):
    """OpenFOAM.com kEqn algebra evaluated on one packed cell."""

    ck: jax.Array
    ce: jax.Array

    def __init__(self, ck: float = CK, ce: float = CE):
        self.ck = jnp.asarray([ck], dtype=jnp.float64)
        self.ce = jnp.asarray([ce], dtype=jnp.float64)

    def __call__(self, features):
        kinetic_energy = jnp.maximum(features[0], 0.0)
        gradient = features[1:10].reshape(3, 3)
        delta = features[10]
        root_k = jnp.sqrt(kinetic_energy)
        nut = self.ck[0] * delta * root_k

        symmetric = 0.5 * (gradient + gradient.T)
        trace = jnp.trace(symmetric)
        dev_two_symmetric = (
            2.0 * symmetric
            - (2.0 / 3.0) * trace * jnp.eye(3, dtype=features.dtype)
        )
        production = nut * jnp.sum(gradient * dev_two_symmetric)
        dissipation = self.ce[0] * root_k / delta
        return jnp.asarray([nut, production, dissipation])


def _run_openfoam(command: str) -> None:
    subprocess.run(
        ["openfoam", "-c", command],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _foam_set(path: Path, entry: str, value: str, *, add: bool = False) -> None:
    mode = "-add" if add else "-set"
    command = (
        f"foamDictionary {shlex.quote(str(path))} "
        f"-entry {shlex.quote(entry)} {mode} {shlex.quote(value)}"
    )
    _run_openfoam(command)


def _clean_source(source: Path, destination: Path, end_time: float) -> None:
    shutil.copytree(source, destination)
    for child in destination.iterdir():
        try:
            numeric = float(child.name)
        except ValueError:
            continue
        if child.is_dir() and numeric != 0.0:
            shutil.rmtree(child)
    shutil.rmtree(destination / "postProcessing", ignore_errors=True)
    control = destination / "system/controlDict"
    _foam_set(control, "startFrom", "startTime")
    _foam_set(control, "startTime", "0")
    _foam_set(control, "endTime", f"{end_time:.16g}")
    _foam_set(control, "writeControl", "runTime")
    _foam_set(control, "writeInterval", f"{end_time:.16g}")
    _foam_set(control, "writePrecision", "16")


def _turbulence_dictionary(model: str) -> str:
    return f"""FoamFile
{{
    format ascii;
    class dictionary;
    location \"constant\";
    object turbulenceProperties;
}}

simulationType LES;

LES
{{
    LESModel {model};
    turbulence on;
    printCoeffs on;
    delta cubeRootVol;

    cubeRootVolCoeffs
    {{
        deltaCoeff 1;
    }}

    {model}Coeffs
    {{
        Ck {CK};
        Ce {CE};
    }}
}}
"""


def _prepare_sources(root: Path, source: Path, end_time: float) -> dict[str, Path]:
    result = {}
    repository = Path(__file__).resolve().parents[2]
    for model in ("Smagorinsky", "kEqn"):
        destination = root / f"source-{model}"
        _clean_source(source, destination, end_time)
        (destination / "constant/turbulenceProperties").write_text(
            _turbulence_dictionary(model), encoding="utf-8"
        )
        if model == "kEqn":
            shutil.copyfile(
                repository / "tools/template/openfoam/k.cavity.in",
                destination / "0/k",
            )
            _foam_set(
                destination / "system/fvSchemes",
                "divSchemes/div(phi,k)",
                "Gauss linear",
                add=True,
            )
        result[model] = destination
    return result


def _export_models(root: Path) -> dict[str, Path]:
    models = root / "models"
    models.mkdir()
    smagorinsky = fno.export.equinox(
        ExactSmagorinsky(),
        path=models / "exact-smagorinsky.fnom",
        inputs={
            "velocity_grad": fno.Tensor.tensor(),
            "filter_width": fno.Tensor.scalar(),
        },
        outputs={"nut": fno.Tensor.scalar()},
    )
    k_equation = fno.export.equinox(
        ExactKEqn(),
        path=models / "exact-keqn.fnom",
        inputs={
            "k": fno.Tensor.scalar(),
            "velocity_grad": fno.Tensor.tensor(),
            "filter_width": fno.Tensor.scalar(),
        },
        outputs={
            "nut": fno.Tensor.scalar(),
            "kProduction": fno.Tensor.scalar(),
            "kDissipationCoeff": fno.Tensor.scalar(),
        },
    )
    return {"Smagorinsky": smagorinsky, "kEqn": k_equation}


def _case(name: str, source: Path, output: Path) -> fno.OpenFOAM.Case:
    return fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd="openfoam",
        shell="zsh",
        application="pimpleFoam",
        ranks=1,
    )


def _closure(model: str, artifact: Path) -> fno.Closure:
    if model == "Smagorinsky":
        return fno.Closure(
            name="nutFjord",
            artifact=artifact,
            inputs={
                "velocity_grad": fno.Math.grad("U"),
                "filter_width": fno.filter_width(),
            },
            outputs={"nut": fno.field("nut")},
        )
    return fno.Closure(
        name="kEqnFjord",
        artifact=artifact,
        inputs={
            "k": fno.field("k"),
            "velocity_grad": fno.Math.grad("U"),
            "filter_width": fno.filter_width(),
        },
        outputs={
            "nut": fno.field("nut"),
            "kProduction": fno.field("kProduction"),
            "kDissipationCoeff": fno.field("kDissipationCoeff"),
        },
    )


def _launch(longship: fno.Longship, timeout: float) -> tuple[fno.Result, float]:
    started = perf_counter()
    result = longship.launch(start_timeout=timeout, verbose=False).stop(
        timeout=timeout
    )
    wall = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"{longship.name} failed; see {result.solver_log}")
    return result, wall


def _solver_timing(path: Path) -> tuple[float, float]:
    matches = re.findall(
        r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s+s\s+"
        r"ClockTime\s*=\s*([0-9.eE+-]+)\s+s",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        return float("nan"), float("nan")
    execution, clock = matches[-1]
    return float(execution), float(clock)


def _reference_scale(result: fno.Result, field: str) -> float:
    values = np.asarray(result.postprocess.field(field, time_idx=-1), dtype=np.float64)
    return float(np.max(np.abs(values)))


def _validate_model(
    model: str,
    source: Path,
    artifact: Path,
    output: Path,
    timeout: float,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    stock, stock_wall = _launch(
        fno.Longship(case=_case(f"{model}-stock", source, output)), timeout
    )
    coupled, coupled_wall = _launch(
        fno.Longship(
            case=_case(f"{model}-FoamNordic", source, output),
            closures=(_closure(model, artifact),),
        ),
        timeout,
    )
    fields = ("U", "p", "nut") if model == "Smagorinsky" else ("U", "p", "nut", "k")
    comparison = fno.Postprocess.compare(
        stock,
        coupled,
        fields=fields,
        time_idx=-1,
        mesh="strict",
        verbose=False,
    )
    passed = True
    for field, metrics in comparison.items():
        scale = _reference_scale(stock, field)
        metrics["reference_max_abs"] = scale
        metrics["relative_linf"] = metrics["max_abs"] / max(scale, np.finfo(float).tiny)
        field_passed = (
            metrics["max_abs"] <= atol + rtol * scale
            and metrics["relative_l2"] <= rtol
        )
        metrics["passed"] = field_passed
        passed = passed and field_passed
    stock_execution, stock_clock = _solver_timing(stock.solver_log)
    coupled_execution, coupled_clock = _solver_timing(coupled.solver_log)
    return {
        "model": model,
        "physical_time": stock.postprocess.times[-1],
        "stock_total_seconds": stock_wall,
        "foamnordic_total_seconds": coupled_wall,
        "stock_openfoam_execution_seconds": stock_execution,
        "foamnordic_openfoam_execution_seconds": coupled_execution,
        "stock_openfoam_clock_seconds": stock_clock,
        "foamnordic_openfoam_clock_seconds": coupled_clock,
        "fields": comparison,
        "passed": passed,
        "stock_run": str(stock.work_dir),
        "foamnordic_run": str(coupled.work_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--end-time", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    arguments = parser.parse_args()
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else Path(tempfile.mkdtemp(prefix="foamnordic-les-parity."))
    )
    output.mkdir(parents=True, exist_ok=True)
    sources = _prepare_sources(
        output,
        arguments.source.expanduser().resolve(),
        arguments.end_time,
    )
    artifacts = _export_models(output)
    results = [
        _validate_model(
            model,
            sources[model],
            artifacts[model],
            output,
            arguments.timeout,
            arguments.rtol,
            arguments.atol,
        )
        for model in ("Smagorinsky", "kEqn")
    ]
    report = {
        "schema": "foamnordic.analytical-les-parity/v1",
        "source": str(arguments.source.expanduser().resolve()),
        "end_time": arguments.end_time,
        "rtol": arguments.rtol,
        "atol": arguments.atol,
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    report_path = output / "analytical-les-parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
