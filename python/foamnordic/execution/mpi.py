"""Local MPI launcher discovery and runtime policy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess

import yaml

from .runtime_paths import RuntimeProfile


PROFILE_NAME = "runtime.yaml"


@dataclass(frozen=True, slots=True)
class MpiPolicy:
    """A local MPI launcher and whether its environment needs isolation."""

    launcher: Path | None
    isolate: bool = False
    source: str = "inherited"


def _executable(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    selected = Path(path).expanduser().resolve()
    if selected.is_file() and os.access(selected, os.X_OK):
        return selected
    return None


def _brew_mpirun() -> Path | None:
    brew = shutil.which("brew")
    if brew is None:
        return None
    try:
        completed = subprocess.run(
            [brew, "--prefix", "open-mpi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _executable(Path(completed.stdout.strip()) / "bin/mpirun")


def _profile_policy(runtime_dir: Path | None) -> MpiPolicy | None:
    if runtime_dir is None:
        return None
    path = runtime_dir / PROFILE_NAME
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mpi = document["mpi"]
    except (OSError, AttributeError, KeyError, TypeError, yaml.YAMLError):
        return None
    launcher = _executable(mpi.get("launcher"))
    if launcher is None:
        return None
    return MpiPolicy(
        launcher=launcher,
        isolate=bool(mpi.get("isolate_launcher_environment", False)),
        source="runtime-profile",
    )


def discover_mpi_policy(
    *,
    wrapper: bool,
    runtime_dir: Path | None = None,
) -> MpiPolicy:
    """Resolve a local launcher without applying HPC scheduler policy."""

    override = os.environ.get("FOAMNORDIC_MPIRUN")
    if override:
        launcher = _executable(override)
        if launcher is None:
            raise RuntimeError(
                f"FOAMNORDIC_MPIRUN is not an executable file: {override}"
            )
        return MpiPolicy(
            launcher=launcher,
            isolate=platform.system() == "Darwin" and wrapper,
            source="environment",
        )

    recorded = _profile_policy(runtime_dir)
    if recorded is not None:
        return recorded

    if platform.system() == "Darwin" and wrapper:
        launcher = _brew_mpirun() or _executable(shutil.which("mpirun"))
        if launcher is not None:
            return MpiPolicy(
                launcher=launcher,
                isolate=True,
                source="macos-auto",
            )
    return MpiPolicy(launcher=None)


def write_runtime_profile(runtime: RuntimeProfile) -> Path:
    """Record the build ABI and reusable local MPI policy."""

    policy = discover_mpi_policy(wrapper=platform.system() == "Darwin")
    document = {
        "schema_version": 1,
        "platform": {
            "system": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "tag": runtime.platform,
        },
        "openfoam": {"abi": runtime.openfoam},
        "mpi": {
            "mode": "external-isolated" if policy.isolate else "inherited",
            "launcher": str(policy.launcher) if policy.launcher else None,
            "isolate_launcher_environment": policy.isolate,
        },
    }
    path = runtime.runtime_dir / PROFILE_NAME
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path
