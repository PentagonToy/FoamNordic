"""Run a two-node OpenFOAM baseline and identity-coupling parity gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil

import numpy as np

import foamnordic as fno


def identity_velocity(velocity):
    return {"velocity": velocity}


def arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate node-local FoamNordic coupling across two Slurm nodes."
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("FOAMNORDIC_SLURM_ACCOUNT"),
    )
    parser.add_argument(
        "--partition",
        default=os.environ.get("FOAMNORDIC_SLURM_PARTITION", "small"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(os.environ["FOAMNORDIC_TEST_ROOT"])
            if "FOAMNORDIC_TEST_ROOT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=(
            repository
            / "tutorials/openfoam_tutorials/incompressible/laminar/lidDrivenCavity"
        ),
    )
    parser.add_argument(
        "--openfoam",
        default=os.environ.get("FOAMNORDIC_OPENFOAM_COMMAND", "openfoam/2512"),
    )
    parser.add_argument("--time", default="00:10:00")
    parser.add_argument("--end-time", type=float, default=0.01)
    options = parser.parse_args()
    if not options.account:
        parser.error("--account or FOAMNORDIC_SLURM_ACCOUNT is required")
    if options.output is None:
        parser.error("--output or FOAMNORDIC_TEST_ROOT is required")
    return options


def stage_case(source: Path, root: Path, end_time: float) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f"source-{timestamp}"
    shutil.copytree(source, destination)
    control = destination / "system/controlDict"
    rendered = control.read_text(encoding="utf-8")
    rendered = re.sub(
        r"(?m)^(\s*endTime\s+)[^;]+;",
        rf"\g<1>{end_time:.12g};",
        rendered,
        count=1,
    )
    rendered = re.sub(
        r"(?m)^(\s*writeInterval\s+)[^;]+;",
        rf"\g<1>{end_time:.12g};",
        rendered,
        count=1,
    )
    control.write_text(rendered, encoding="utf-8")
    return destination


def make_case(name: str, source: Path, output: Path, openfoam: str) -> fno.OpenFOAM.Case:
    case = fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd=openfoam,
        shell="bash",
        application="pimpleFoam",
        ranks=2,
    )
    case.initialize(ranks=2, mesh="blockMesh", validate_mesh=True)
    return case


def compare(baseline, coupled) -> None:
    for field in ("U", "p"):
        reference = baseline.postprocess.field(field, time_idx=-1)
        candidate = coupled.postprocess.field(field, time_idx=-1)
        maximum_error = float(np.max(np.abs(candidate - reference)))
        print(f"[FoamNordic] Two-node parity: {field} max abs = {maximum_error:.6e}")
        if not np.allclose(candidate, reference, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError(f"two-node identity parity failed for {field}")


def require_success(label: str, result) -> None:
    if result.success:
        return
    print(f"[FoamNordic] {label} failed before parity comparison.")
    for path in (
        result.work_dir / ".foamnordic/submission.log",
        result.longship_log,
        result.host_log,
        result.solver_log,
    ):
        if path.is_file():
            print(f"\n[FoamNordic] {path}")
            print(path.read_text(encoding="utf-8", errors="replace")[-8000:])
    raise RuntimeError(f"two-node {label} run failed")


def main() -> int:
    options = arguments()
    output = options.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = stage_case(options.case.expanduser().resolve(), output, options.end_time)

    baseline_openfoam = fno.Slurm.openfoam(
        nodes=2,
        ntasks=2,
        cpus_per_task=2,
    )
    coupled_openfoam = fno.Slurm.openfoam(
        nodes=2,
        ntasks=2,
        cpus_per_task=1,
        mem_per_cpu="2G",
    )
    baseline_scheduler = fno.Slurm(
        account=options.account,
        partition=options.partition,
        time=options.time,
        openfoam=baseline_openfoam,
    )
    coupled_scheduler = fno.Slurm(
        account=options.account,
        partition=options.partition,
        time=options.time,
        openfoam=coupled_openfoam,
        model=fno.Slurm.model(cpus_per_task=1, mem_per_cpu="1G"),
    )

    baseline_case = make_case("multiNodeBaseline", source, output, options.openfoam)
    baseline_run = fno.Longship(
        case=baseline_case,
        scheduler=baseline_scheduler,
    ).launch(start_timeout=900)
    baseline = baseline_run.stop(timeout=1800, progress=True)
    baseline.summary(style="compact")
    require_success("baseline", baseline)

    transform = fno.Transform(
        name="identityVelocity",
        operator=fno.Operator.function(identity_velocity),
        inputs={"velocity": fno.Field("U")},
        outputs={"velocity": fno.Field("U")},
        at="time_step_start",
    )
    coupled_case = make_case("multiNodeIdentity", source, output, options.openfoam)
    coupled_run = fno.Longship(
        case=coupled_case,
        scheduler=coupled_scheduler,
        transforms=(transform,),
    ).launch(start_timeout=900)
    coupled = coupled_run.stop(timeout=1800, progress=True)
    coupled.summary(style="compact")
    require_success("identity", coupled)

    compare(baseline, coupled)
    print("[FoamNordic] Two-node Longship gate: PASS")
    print(f"[FoamNordic] Baseline: {baseline.work_dir}")
    print(f"[FoamNordic] Coupled: {coupled.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
