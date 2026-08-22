"""Command-line entry point for FoamNordic installation and runtime tasks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from time import monotonic
from typing import Iterator, Sequence, TextIO

from ._managed import generated_kind, mark_generated


_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class _Terminal:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.interactive = stream.isatty()
        self.color = self.interactive and os.environ.get("NO_COLOR") is None
        self.green = "\033[1;32m" if self.color else ""
        self.blue = "\033[1;34m" if self.color else ""
        self.red = "\033[1;31m" if self.color else ""
        self.dim = "\033[2m" if self.color else ""
        self.reset = "\033[0m" if self.color else ""
        self.clear = "\033[K" if self.interactive else ""

    def section(self, title: str) -> None:
        rule = "=" * 66
        print(f"\n{rule}\n {title}\n{rule}\n", file=self.stream)

    @contextmanager
    def step(self, index: int, total: int, description: str) -> Iterator[None]:
        started = monotonic()
        stop = threading.Event()

        def animate() -> None:
            frame = 0
            while not stop.wait(0.1):
                elapsed = _duration(monotonic() - started)
                marker = _FRAMES[frame % len(_FRAMES)]
                print(
                    f"\r{self.blue}{marker}{self.reset} "
                    f"{self.green}[Step {index}/{total}]{self.reset} "
                    f"{description} {self.dim}[{elapsed}]{self.reset}{self.clear}",
                    end="",
                    file=self.stream,
                    flush=True,
                )
                frame += 1

        thread = threading.Thread(target=animate, daemon=True)
        if self.interactive:
            thread.start()
        else:
            print(f"[Step {index}/{total}] {description} ...", file=self.stream)

        try:
            yield
        except Exception:
            stop.set()
            if thread.is_alive():
                thread.join()
            elapsed = _duration(monotonic() - started)
            prefix = "\r" if self.interactive else ""
            print(
                f"{prefix}{self.red}✗{self.reset} [Step {index}/{total}] "
                f"{description} {self.dim}[{elapsed}]{self.reset}{self.clear}",
                file=self.stream,
            )
            raise
        else:
            stop.set()
            if thread.is_alive():
                thread.join()
            elapsed = _duration(monotonic() - started)
            prefix = "\r" if self.interactive else ""
            print(
                f"{prefix}{self.green}✓{self.reset} [Step {index}/{total}] "
                f"{description} {self.dim}[{elapsed}]{self.reset}{self.clear}",
                file=self.stream,
            )


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _source_root(explicit: Path | None) -> Path | None:
    candidates = [explicit] if explicit is not None else [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.expanduser().resolve()
        if (root / "CMakeLists.txt").is_file() and (
            root / "src/foamnordic/runtime"
        ).is_dir():
            return root
    if explicit is not None:
        raise FileNotFoundError(f"FoamNordic source tree was not found: {explicit}")
    return None


def _default_jobs() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value and value.isdigit() and int(value) > 0:
        return int(value)
    return min(os.cpu_count() or 1, 8)


def _verify_packaged_runtime() -> None:
    from . import _native

    request = _native.LongshipRequest()
    plan = _native.plan_longship(request)
    if not plan.host_starts_first or not plan.fail_together:
        raise RuntimeError("packaged native runtime violated lifecycle invariants")


def _directories(stream: TextIO) -> int:
    terminal = _Terminal(stream)
    terminal.section("FoamNordic directories")
    package = Path(__file__).resolve().parent
    source = _source_root(None)
    try:
        from . import _native

        native_file = Path(_native.__file__).resolve() if _native.__file__ else None
    except ImportError:
        native_file = None

    rows = (
        ("Python environment", Path(sys.prefix).resolve()),
        ("Python package", package),
        ("Native module", native_file or "not installed"),
        ("Source tree", source or "not detected"),
        ("Build cache", Path.home() / ".cache/foamnordic/build"),
        ("Native SDK", Path.home() / ".local/share/foamnordic/native"),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}", file=stream)
    return 0


def _run_command(command: Sequence[str], log: TextIO, *, dry_run: bool) -> None:
    if dry_run:
        print(f"$ {shlex.join(command)}", file=log)
        return
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _build(args: argparse.Namespace, stream: TextIO) -> int:
    terminal = _Terminal(stream)
    terminal.section("FoamNordic native build")
    source = _source_root(args.source)

    if source is None:
        with terminal.step(1, 1, "Verify packaged native runtime"):
            _verify_packaged_runtime()
        print("\nFoamNordic is ready. This wheel requires no native core rebuild.", file=stream)
        print("OpenFOAM integration will use the case environment.", file=stream)
        return 0

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake is required to build the native SDK")

    build_dir = (args.build_dir or Path.home() / ".cache/foamnordic/build").expanduser()
    prefix = (args.prefix or Path.home() / ".local/share/foamnordic/native").expanduser()
    log_path = build_dir / "foamnordic-build.log"
    build_was_present = build_dir.exists()
    prefix_was_present = prefix.exists()
    if not args.dry_run:
        build_dir.mkdir(parents=True, exist_ok=True)
        prefix.mkdir(parents=True, exist_ok=True)
        if not build_was_present:
            mark_generated(build_dir, kind="build")
        if not prefix_was_present:
            mark_generated(prefix, kind="native-prefix")

    commands = (
        (
            "Configure native SDK",
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DFOAMNORDIC_TESTS=OFF",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
            ],
        ),
        (
            "Build native runtime",
            [
                cmake,
                "--build",
                str(build_dir),
                "--target",
                "foamnordic_runtime",
                "--parallel",
                str(args.jobs),
            ],
        ),
        (
            "Install C++ development SDK",
            [
                cmake,
                "--install",
                str(build_dir),
                "--component",
                "Development",
            ],
        ),
    )

    if args.dry_run:
        log: TextIO = stream
        for index, (description, command) in enumerate(commands, start=1):
            with terminal.step(index, len(commands), description):
                _run_command(command, log, dry_run=True)
    else:
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for index, (description, command) in enumerate(commands, start=1):
                    with terminal.step(index, len(commands), description):
                        _run_command(command, log, dry_run=False)
        except Exception:
            print(f"Build log: {log_path}", file=stream)
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-60:]), file=stream)
            raise

    print(f"\nNative SDK: {prefix}", file=stream)
    if not args.dry_run:
        cache = build_dir / "CMakeCache.txt"
        package = prefix / "lib/cmake/FoamNordic/FoamNordicConfig.cmake"
        cache_text = (
            cache.read_text(encoding="utf-8", errors="ignore")
            if cache.is_file()
            else ""
        )
        if generated_kind(build_dir) is None and "FoamNordic" in cache_text:
            mark_generated(build_dir, kind="build")
        if generated_kind(prefix) is None and package.is_file():
            mark_generated(prefix, kind="native-prefix")
        print(f"Build log:  {log_path}", file=stream)
    return 0


def _clobber_targets(args: argparse.Namespace) -> list[Path]:
    candidates = [
        (Path.home() / ".cache/foamnordic/build").resolve(),
        (Path.home() / ".local/share/foamnordic/native").resolve(),
    ]
    for workspace in args.workspace:
        runs = workspace.expanduser().resolve() / "runs"
        if runs.is_dir():
            candidates.extend(
                marker.parent
                for marker in runs.glob("*/.foamnordic-generated.json")
            )
    candidates.extend(path.expanduser().resolve() for path in args.virtual)
    return list(dict.fromkeys(candidates))


def _clobber(args: argparse.Namespace, stream: TextIO) -> int:
    terminal = _Terminal(stream)
    terminal.section("FoamNordic generated asset cleanup")
    removed = 0
    for target in _clobber_targets(args):
        kind = generated_kind(target)
        if kind is None:
            if target.exists():
                print(f"skip (not FoamNordic-owned): {target}", file=stream)
            continue
        if target in {Path("/").resolve(), Path.home().resolve()}:
            raise RuntimeError(f"refusing broad cleanup target: {target}")
        action = "would remove" if args.dry_run else "remove"
        print(f"{action} [{kind}]: {target}", file=stream)
        if not args.dry_run:
            shutil.rmtree(target)
        removed += 1
    if removed == 0:
        print("Nothing to clobber. Unmarked directories were preserved.", file=stream)
    elif args.dry_run:
        noun = "directory" if removed == 1 else "directories"
        print(f"\n{removed} marked {noun} selected.", file=stream)
    else:
        noun = "directory" if removed == 1 else "directories"
        print(f"\nRemoved {removed} FoamNordic-generated {noun}.", file=stream)
    return 0


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="foamnordic",
        description="Build and run native OpenFOAM closure workloads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "dir",
        help="show the active package, environment, and native directories",
        description="Show the directories used by the active FoamNordic installation.",
    )
    build = subcommands.add_parser(
        "build",
        help="verify a binary wheel or build the native C++ SDK",
        description=(
            "Verify the native runtime bundled in a binary wheel. When run in "
            "a FoamNordic source tree, build and install its C++ SDK instead."
        ),
    )
    build.add_argument("--source", type=Path, help="FoamNordic source tree")
    build.add_argument("--build-dir", type=Path, help="CMake build directory")
    build.add_argument("--prefix", type=Path, help="native SDK install prefix")
    build.add_argument("--jobs", type=int, default=_default_jobs(), help="parallel jobs")
    build.add_argument("--dry-run", action="store_true", help="show commands only")
    clobber = subcommands.add_parser(
        "clobber",
        help="remove only marker-owned build and run assets",
        description=(
            "Remove FoamNordic-generated build assets. Source cases and unmarked "
            "directories are always preserved. Add --workspace to include marked "
            "run directories below WORKSPACE/runs."
        ),
    )
    clobber.add_argument(
        "--workspace",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="also remove marked runs below PATH/runs (repeatable)",
    )
    clobber.add_argument(
        "--virtual",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="also remove an explicitly selected marker-owned virtual environment",
    )
    clobber.add_argument("--dry-run", action="store_true", help="show removals only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "dir":
        return _directories(sys.stdout)
    if args.command == "build":
        if args.jobs <= 0:
            parser.error("--jobs must be positive")
        try:
            return _build(args, sys.stdout)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    if args.command == "clobber":
        try:
            return _clobber(args, sys.stdout)
        except (OSError, RuntimeError) as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
