"""Private shell rendering shared by case and scheduler backends."""

from __future__ import annotations

import shlex
from collections.abc import Sequence


def quote_command(values: Sequence[object]) -> str:
    return shlex.join([str(value) for value in values])


def toolchain_shell(toolchain: object, command: str) -> tuple[str, ...]:
    prefix = toolchain.command
    if getattr(toolchain, "wrapper", False):
        return (
            str(toolchain.shell),
            "-lc",
            "exec " + quote_command((*shlex.split(prefix), "-c", command)),
        )
    script = f"{prefix}; {command}" if prefix else command
    return (str(toolchain.shell), "-lc", script)
