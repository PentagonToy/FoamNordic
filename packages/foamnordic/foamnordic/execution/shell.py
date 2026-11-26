"""Private shell rendering shared by case and scheduler backends."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path


def quote_command(values: Sequence[object]) -> str:
    return shlex.join([str(value) for value in values])


def toolchain_shell(toolchain: object, command: str) -> tuple[str, ...]:
    prefix = toolchain.command
    if getattr(toolchain, "wrapper", False):
        launcher = shlex.split(prefix)
        if Path(launcher[0]).name == "openfoam-14":
            arguments = (*launcher, "--", str(toolchain.shell), "-c", command)
        else:
            arguments = (*launcher, "-c", command)
        return str(toolchain.shell), "-lc", "exec " + quote_command(arguments)
    script = f"{prefix}; {command}" if prefix else command
    return (str(toolchain.shell), "-lc", script)
