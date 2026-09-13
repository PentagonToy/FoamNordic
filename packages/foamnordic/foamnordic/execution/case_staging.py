"""Own isolated OpenFOAM case staging and preparation side effects."""

from __future__ import annotations

import re
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.managed import mark_generated
from ..core.plan import CompiledPlan
from .run_files import initialize_sailing_log, sailing_paths
from .shell import toolchain_shell

if TYPE_CHECKING:
    from ..core.spec import Longship


def stage_case(longship: Longship, plan: CompiledPlan) -> tuple[Path, Path]:
    """Allocate one generated run and copy its source case in isolation."""

    workspace = longship.case.run_dir.expanduser().resolve()
    runs = workspace / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_name = re.sub(r"[^A-Za-z0-9-]+", "-", longship.name).strip("-")
    run_name = run_name or "FoamNordic"
    for _ in range(100):
        work_dir = runs / f"{run_name}-preparing-{secrets.token_hex(4)}"
        try:
            work_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Cannot allocate a unique FoamNordic run directory")

    mark_generated(
        work_dir,
        kind="run",
        metadata={"plan": plan.as_dict(), "plan_digest": plan.digest},
    )
    case_dir = work_dir / "case"
    shutil.copytree(longship.case.case_dir.expanduser().resolve(), case_dir)
    if not (case_dir / "0").exists() and (case_dir / "0.orig").is_dir():
        shutil.copytree(case_dir / "0.orig", case_dir / "0")
    return work_dir, case_dir


def prepare_staged_case(
    longship: Longship,
    work_dir: Path,
    commands: list[str],
    *,
    verbose: bool = False,
    mesh_commands: bool = False,
) -> None:
    """Execute the complete preparation transaction and retain its log."""

    if verbose and mesh_commands:
        if longship.case._mesh == "blockMesh":
            print(f"[FoamNordic] Preparing mesh with blockMesh: {longship.name}")
        else:
            print(
                "[FoamNordic] Validating existing mesh with checkMesh: "
                f"{longship.name}"
            )
    sailing_log, _, _ = sailing_paths(work_dir, longship.name)
    initialize_sailing_log(sailing_log, longship.name)
    with sailing_log.open("a", encoding="utf-8") as stream:
        stream.write("[FoamNordic] Preparing isolated OpenFOAM case\n")
    try:
        with sailing_log.open("ab") as stream:
            subprocess.run(
                toolchain_shell(longship.case._toolchain, " && ".join(commands)),
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"OpenFOAM case preparation failed; inspect {sailing_log}"
        ) from error
    if verbose and mesh_commands:
        print(f"[FoamNordic] Mesh is ready: {longship.name}")


def verify_decomposition(case_dir: Path, ranks: int) -> None:
    """Require exactly one processor directory for every requested rank."""

    if ranks <= 1:
        return
    expected = {f"processor{rank}" for rank in range(ranks)}
    actual = {
        path.name
        for path in case_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"processor\d+", path.name)
    }
    if actual != expected:
        raise RuntimeError(
            "OpenFOAM decomposition does not match Case.ranks: "
            f"expected {sorted(expected)}, found {sorted(actual)}"
        )


__all__ = ["prepare_staged_case", "stage_case", "verify_decomposition"]
