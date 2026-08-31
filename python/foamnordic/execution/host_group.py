"""Fail-together supervisor for several independent resident field programs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _ready_path(value: str, suffix: str) -> Path:
    return Path(value + suffix)


def _rewrite_ready_files(command: list[str], suffix: str) -> list[str]:
    rewritten = list(command)
    for index, argument in enumerate(rewritten[:-1]):
        if argument == "--ready-file":
            rewritten[index + 1] += suffix
    return rewritten


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        process.wait()


def main(path: str) -> int:
    configuration = json.loads(Path(path).read_text(encoding="utf-8"))
    suffix = os.environ.get("FOAMNORDIC_READY_SUFFIX", "")
    commands = [
        _rewrite_ready_files(command, suffix)
        for command in configuration["commands"]
    ]
    ready_files = [
        _ready_path(value, suffix) for value in configuration["ready_files"]
    ]
    aggregate = _ready_path(configuration["aggregate_ready"], suffix)
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        processes = [subprocess.Popen(command) for command in commands]
        published = False
        while True:
            if stopping:
                _terminate(processes)
                return 130
            statuses = [process.poll() for process in processes]
            failed = next((status for status in statuses if status not in {None, 0}), None)
            if failed is not None:
                _terminate(processes)
                return int(failed)
            if not published and all(path.is_file() for path in ready_files):
                aggregate.parent.mkdir(parents=True, exist_ok=True)
                aggregate.touch(exist_ok=False)
                published = True
            if all(status is not None for status in statuses):
                return 0
            time.sleep(0.01)
    finally:
        aggregate.unlink(missing_ok=True)
        for path in ready_files:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m foamnordic.execution.host_group CONFIG.json"
        )
    raise SystemExit(main(sys.argv[1]))
