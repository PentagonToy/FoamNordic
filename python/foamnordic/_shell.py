"""Private shell rendering shared by case and scheduler backends."""

from __future__ import annotations

import shlex
from typing import Sequence


def quote_command(values: Sequence[object]) -> str:
    return shlex.join([str(value) for value in values])


def toolchain_shell(toolchain: object, command: str) -> tuple[str, ...]:
    prefix = getattr(toolchain, "command")
    script = f"{prefix}; {command}" if prefix else command
    return (str(getattr(toolchain, "shell")), "-lc", script)
