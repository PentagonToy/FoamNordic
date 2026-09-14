"""Measure rank-weighted resident model work on a periodic channel case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np

import foamnordic as fno


def rank_weighted_identity(velocity, *, rank):
    """Return velocity unchanged after deterministic rank-weighted work."""

    values = np.asarray(velocity)
    scratch = np.asarray(values[:, 0], dtype=np.float64)
    for _ in range(16 * (rank % 4 + 1)):
        scratch = np.sqrt(scratch * scratch + 1.0)
    if not np.all(np.isfinite(scratch)):
        raise FloatingPointError("rank-weighted benchmark produced a non-finite value")
    return {"velocity": values}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--account",
        default=os.environ.get("FOAMNORDIC_SLURM_ACCOUNT"),
    )
    parser.add_argument("--partition", default="test")
    parser.add_argument("--time", default="00:15:00")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--model-cpus", type=int, default=1)
    parser.add_argument("--simple-n", type=int, nargs=3)
    parser.add_argument("--openfoam", default="openfoam/2512")
    options = parser.parse_args()
    if not options.account:
        parser.error("--account or FOAMNORDIC_SLURM_ACCOUNT is required")
    if options.model_cpus < 1:
        parser.error("--model-cpus must be positive")
    if options.simple_n is None:
        defaults = {4: (1, 2, 2), 8: (2, 2, 2), 16: (2, 4, 2)}
        try:
            options.simple_n = defaults[options.ranks]
        except KeyError:
            parser.error("--simple-n is required unless --ranks is 4, 8, or 16")
    if np.prod(options.simple_n) != options.ranks:
        parser.error("the product of --simple-n must equal --ranks")
    return options


def stage_source(source: Path, output: Path, ranks: int, simple_n) -> Path:
    """Copy the case and install an explicit rank-compatible decomposition."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / f"source-r{ranks}-{stamp}"
    shutil.copytree(source, destination)
    nx, ny, nz = simple_n
    (destination / "system/decomposeParDict").write_text(
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
        f"    n ({nx} {ny} {nz});\n"
        "    delta 0.001;\n"
        "}\n",
        encoding="utf-8",
    )
    return destination


def case(
    name: str,
    source: Path,
    output: Path,
    openfoam: str,
    ranks: int,
) -> fno.OpenFOAM.Case:
    declared = fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd=openfoam,
        shell="bash",
        application="pimpleFoam",
        ranks=ranks,
    )
    return declared.initialize(ranks=ranks, mesh=None, validate_mesh=True)


def scheduler(options: argparse.Namespace, *, coupled: bool) -> fno.Slurm:
    model = (
        fno.Slurm.model(cpus_per_task=options.model_cpus, mem_per_cpu="2G")
        if coupled
        else None
    )
    return fno.Slurm(
        account=options.account,
        partition=options.partition,
        time=options.time,
        openfoam=fno.Slurm.openfoam(
            nodes=1,
            ntasks=options.ranks,
            cpus_per_task=1,
            mem_per_cpu="2G",
        ),
        model=model,
    )


def completed(label: str, run: fno.Run):
    result = run.stop(timeout=1800.0, progress=True)
    result.summary(style="compact")
    if not result.success:
        raise RuntimeError(f"{label} failed; see {result.solver_log}")
    return result


def main() -> int:
    options = arguments()
    output = options.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = stage_source(
        options.case.expanduser().resolve(),
        output,
        options.ranks,
        options.simple_n,
    )

    baseline = completed(
        "baseline",
        fno.Longship(
            case=case(
                "channelBaseline",
                source,
                output,
                options.openfoam,
                options.ranks,
            ),
            scheduler=scheduler(options, coupled=False),
        ).launch(start_timeout=900.0),
    )

    transform = fno.Transform(
        name="rankWeightedIdentity",
        operator=fno.Operator.function(rank_weighted_identity),
        inputs={"velocity": fno.Field("U")},
        outputs={"velocity": fno.Field("U")},
        at="time_step_start",
    )
    if transform.execution_lane != "resident":
        raise RuntimeError("rank-weighted benchmark must use resident execution")
    coupled = completed(
        "rank-weighted",
        fno.Longship(
            case=case(
                "channelRankWeighted",
                source,
                output,
                options.openfoam,
                options.ranks,
            ),
            transforms=(transform,),
            scheduler=scheduler(options, coupled=True),
        ).launch(start_timeout=900.0, profile_every=25),
    )

    comparison = fno.Postprocess.compare(
        baseline,
        coupled,
        fields=("U", "p"),
        time_idx=-1,
        mesh="strict",
        verbose=False,
    )
    for field, metrics in comparison.items():
        maximum = float(metrics["max_abs"])
        print(f"[FoamNordic] Identity parity: {field} max abs = {maximum:.6e}")
        if maximum != 0.0:
            raise RuntimeError(f"rank-weighted identity changed {field}")

    print(f"[FoamNordic] Baseline: {baseline.work_dir}")
    print(f"[FoamNordic] Rank-weighted: {coupled.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
