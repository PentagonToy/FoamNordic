"""Platform and OpenFOAM ABI-specific native runtime locations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

from .shell import toolchain_shell


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """The active platform/OpenFOAM ABI and its managed directories."""

    platform: str
    openfoam: str
    build_dir: Path
    runtime_dir: Path


def _clean(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{system}-{machine}"


def _command_output(command: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""


def _abi(version: str, options: str) -> str | None:
    normalized = version.strip().lstrip("v")
    if not normalized:
        return None
    suffix = _clean(options) or "default"
    return f"openfoam-v{_clean(normalized)}-{suffix}"


def openfoam_abi(*, required: bool = True) -> str | None:
    """Identify the currently loaded OpenFOAM runtime without hard-coded sites."""

    version = os.environ.get("WM_PROJECT_VERSION", "").lstrip("v")
    if not version and shutil.which("foamVersion"):
        version = _command_output(("foamVersion",)).lstrip("v")
    options = os.environ.get("WM_OPTIONS", "")
    if not options:
        options = os.environ.get("FOAM_API", "")
    if version and shutil.which("wmake"):
        return _abi(version, options)
    if required:
        raise RuntimeError(
            "an OpenFOAM environment is required; load its module on HPC or "
            "enter the OpenFOAM shell on macOS before running this command"
        )
    return None


def openfoam_abi_for_toolchain(
    toolchain: object,
    *,
    required: bool = True,
) -> str | None:
    """Inspect the environment declared by ``Case.of_cmd`` in isolation."""

    probe = (
        "command -v wmake >/dev/null 2>&1 || exit 91; "
        "printf '__FOAMNORDIC_VERSION__=%s\\n' \"${WM_PROJECT_VERSION:-}\"; "
        "printf '__FOAMNORDIC_OPTIONS__=%s\\n' \"${WM_OPTIONS:-}\""
    )
    try:
        completed = subprocess.run(
            toolchain_shell(toolchain, probe),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    values: dict[str, str] = {}
    if completed is not None and completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if line.startswith("__FOAMNORDIC_") and "=" in line:
                name, value = line.split("=", 1)
                values[name] = value.strip()
    selected = _abi(
        values.get("__FOAMNORDIC_VERSION__", ""),
        values.get("__FOAMNORDIC_OPTIONS__", ""),
    )
    if selected is not None:
        return selected
    if required:
        raise RuntimeError(
            "Case.of_cmd did not expose a usable OpenFOAM environment; verify "
            "that it loads wmake, WM_PROJECT_VERSION, and WM_OPTIONS"
        )
    return None


def runtime_directory(abi: str) -> Path:
    return (
        Path.home()
        / ".local/share/foamnordic/runtime"
        / platform_tag()
        / abi
    ).resolve()


def profile(
    *,
    build_dir: Path | None = None,
    runtime_dir: Path | None = None,
    required: bool = True,
) -> RuntimeProfile | None:
    abi = openfoam_abi(required=required)
    if abi is None:
        return None
    platform_value = platform_tag()
    build = (
        Path(build_dir).expanduser()
        if build_dir is not None
        else Path.home() / ".cache/foamnordic/build" / platform_value / abi
    )
    runtime = (
        Path(runtime_dir).expanduser()
        if runtime_dir is not None
        else runtime_directory(abi)
    )
    return RuntimeProfile(
        platform=platform_value,
        openfoam=abi,
        build_dir=build.resolve(),
        runtime_dir=runtime.resolve(),
    )


def active_runtime_candidates() -> tuple[Path, ...]:
    selected = profile(required=False)
    return () if selected is None else (selected.runtime_dir,)


def toolchain_runtime_candidates(toolchain: object) -> tuple[Path, ...]:
    abi = openfoam_abi_for_toolchain(toolchain, required=False)
    return () if abi is None else (runtime_directory(abi),)
