"""Specialize one attached ClosureHost command for its Slurm node."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _node_index(nodes: int) -> int:
    value = os.environ.get("SLURM_PROCID")
    if value is None:
        raise RuntimeError("multi-node ClosureHost requires SLURM_PROCID")
    try:
        index = int(value)
    except ValueError as error:
        raise RuntimeError("SLURM_PROCID must be an integer") from error
    if index < 0 or index >= nodes:
        raise RuntimeError(
            f"ClosureHost node index {index} is outside the {nodes}-node allocation"
        )
    return index


def _rewrite_ready_file(command: list[str], suffix: str) -> list[str]:
    rewritten = list(command)
    for index, argument in enumerate(rewritten[:-1]):
        if argument == "--ready-file":
            rewritten[index + 1] += suffix
    return rewritten


def main(path: str) -> int:
    configuration = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = int(configuration["nodes"])
    index = _node_index(nodes)
    suffix = f".node{index}"
    command = _rewrite_ready_file(configuration["command"], suffix)
    environment = dict(os.environ)
    environment["FOAMNORDIC_READY_SUFFIX"] = suffix
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m foamnordic.execution.node_host CONFIG.json"
        )
    raise SystemExit(main(sys.argv[1]))
