"""Run several field programs inside one node-local ModelHost process."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from bindnordic.models.runtime import configure_model_threads

from .. import _native
from .resident import run_program


def _ready_path(value: str, suffix: str) -> Path:
    return Path(value + suffix)


def _rewrite_ready_files(arguments: list[str], suffix: str) -> list[str]:
    rewritten = list(arguments)
    for index, argument in enumerate(rewritten[:-1]):
        if argument == "--ready-file":
            rewritten[index + 1] += suffix
    return rewritten


def _socket_paths(programs: list[list[str]]) -> tuple[Path, ...]:
    return tuple(
        Path(argument.removeprefix("unix://"))
        for arguments in programs
        for argument in arguments
        if argument.startswith("unix://")
    )


def _cleanup(paths: tuple[Path, ...]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def main(path: str) -> int:
    configuration = json.loads(Path(path).read_text(encoding="utf-8"))
    for plugin in configuration.get("connector_plugins", ()):
        _native._load_connector_plugin(str(plugin))
    suffix = os.environ.get("FOAMNORDIC_READY_SUFFIX", "")
    programs = [
        _rewrite_ready_files(arguments, suffix)
        for arguments in configuration["programs"]
    ]
    ready_files = [
        _ready_path(value, suffix) for value in configuration["ready_files"]
    ]
    if not programs or len(programs) != len(ready_files):
        raise ValueError(
            "ModelHost requires one readiness marker for each field program"
        )
    aggregate = _ready_path(configuration["aggregate_ready"], suffix)
    cleanup_paths = (aggregate, *ready_files, *_socket_paths(programs))
    capacity_units = configure_model_threads(int(configuration["capacity_units"]))
    capacity = _native._ExecutionCapacity(capacity_units)
    threads: list[threading.Thread] = []
    failures: list[BaseException] = []
    completed = 0
    state = threading.Lock()
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    def serve(arguments: list[str]) -> None:
        nonlocal completed
        try:
            run_program(arguments, capacity, configure_threads=False)
        # A worker thread must report SystemExit and KeyboardInterrupt as well as
        # ordinary exceptions so that the group can fail together.
        except BaseException as error:  # noqa: BLE001
            with state:
                failures.append(error)
            return
        with state:
            completed += 1

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        threads = [
            threading.Thread(target=serve, args=(arguments,), daemon=True)
            for arguments in programs
        ]
        for thread in threads:
            thread.start()
        published = False
        while True:
            with state:
                failure = failures[0] if failures else None
                all_completed = completed == len(threads)
            if stopping:
                _cleanup(cleanup_paths)
                os._exit(130)
            if failure is not None:
                print(
                    f"[FoamNordic] ModelHost program failed: {failure}",
                    file=sys.stderr,
                    flush=True,
                )
                _cleanup(cleanup_paths)
                os._exit(1)
            if not published and all(path.is_file() for path in ready_files):
                aggregate.parent.mkdir(parents=True, exist_ok=True)
                aggregate.touch(exist_ok=False)
                published = True
            if all_completed:
                for thread in threads:
                    thread.join()
                return 0
            time.sleep(0.01)
    finally:
        _cleanup(cleanup_paths)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m foamnordic.execution.host_group CONFIG.json"
        )
    raise SystemExit(main(sys.argv[1]))
