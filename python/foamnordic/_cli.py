"""Command-line entry point for FoamNordic installation and runtime tasks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from time import monotonic
from typing import Iterator, Sequence, TextIO

from .build.onnxruntime import VERSION as ONNXRUNTIME_VERSION
from .build.onnxruntime import resolve as resolve_onnxruntime
from .core.managed import generated_kind, mark_generated
from .execution.mpi import write_runtime_profile
from .execution.runtime_paths import active_runtime_candidates, platform_tag, profile


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
    candidates = (
        [explicit]
        if explicit is not None
        else [
            Path.cwd(),
            *Path.cwd().parents,
            Path(__file__).resolve().parent / "buildkit",
            Path(__file__).resolve().parents[2],
        ]
    )
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


def _installed_openfoam_library(runtime: Path) -> Path:
    candidates = (
        *sorted(runtime.glob("lib/libfoamnordicOpenFOAM-*.dylib")),
        runtime / "lib/libfoamnordicOpenFOAM.so",
        runtime / "lib/libfoamnordicOpenFOAM.dylib",
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        raise RuntimeError(
            "the OpenFOAM integration is not installed for the active ABI; "
            "run foamnordic build first"
        )
    return library.resolve()


def _directories(args: argparse.Namespace, stream: TextIO) -> int:
    package = Path(__file__).resolve().parent
    source = _source_root(None)
    try:
        from . import _native

        native_file = Path(_native.__file__).resolve() if _native.__file__ else None
    except ImportError:
        native_file = None

    selected = profile(required=False)
    if args.runtime or args.include or args.openfoam_library:
        if selected is None:
            raise RuntimeError(
                "an active OpenFOAM environment is required for SDK path lookup"
            )
        if args.runtime:
            print(selected.runtime_dir, file=stream)
        elif args.include:
            print(selected.runtime_dir / "include", file=stream)
        else:
            print(_installed_openfoam_library(selected.runtime_dir), file=stream)
        return 0

    terminal = _Terminal(stream)
    terminal.section("FoamNordic directories")
    rows = (
        ("Python environment", Path(sys.prefix).resolve()),
        ("Python package", package),
        ("Native module", native_file or "not installed"),
        ("Build source", source or "not detected"),
        (
            "OpenFOAM ABI",
            selected.openfoam if selected is not None else "not loaded",
        ),
        (
            "Build cache",
            selected.build_dir
            if selected is not None
            else Path.home() / ".cache/foamnordic/build/<platform>/<openfoam-abi>",
        ),
        (
            "Native runtime",
            selected.runtime_dir
            if selected is not None
            else Path.home()
            / ".local/share/foamnordic/runtime/<platform>/<openfoam-abi>",
        ),
        (
            "Runtime profile",
            selected.runtime_dir / "runtime.yaml"
            if selected is not None
            else "not generated",
        ),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}", file=stream)
    return 0


def _doctor_checks() -> list[tuple[str, str, str]]:
    """Run fast, read-only checks for the active FoamNordic environment."""

    from . import __version__

    checks: list[tuple[str, str, str]] = []
    checks.append(
        (
            "PASS",
            "FoamNordic",
            f"{__version__} at {Path(__file__).resolve().parent}",
        )
    )

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        (
            "PASS" if python_supported else "FAIL",
            "Python",
            f"{python_version} at {Path(sys.executable).resolve()}",
        )
    )

    selected_platform = platform_tag()
    supported_platforms = {"darwin-aarch64", "linux-aarch64", "linux-x86_64"}
    checks.append(
        (
            "PASS" if selected_platform in supported_platforms else "WARN",
            "Platform",
            selected_platform,
        )
    )

    try:
        from . import _native

        native_file = Path(_native.__file__).resolve() if _native.__file__ else None
    except ImportError as error:
        checks.append(("FAIL", "Native extension", str(error)))
    else:
        required_native = ("plan_longship", "read_model_manifest", "write_model_bundle")
        missing_native = [
            name for name in required_native if not hasattr(_native, name)
        ]
        checks.append(
            (
                "FAIL" if missing_native else "PASS",
                "Native extension",
                f"missing API: {', '.join(missing_native)}"
                if missing_native
                else str(native_file or "loaded"),
            )
        )

    selected = profile(required=False)
    wmake = shutil.which("wmake")
    if selected is None:
        checks.append(("WARN", "OpenFOAM", "not loaded; wmake and ABI are unavailable"))
    else:
        checks.append(("PASS", "OpenFOAM", selected.openfoam))

    compiler_name = os.environ.get("CXX", "c++")
    compiler = shutil.which(compiler_name)
    checks.append(
        (
            "PASS"
            if compiler is not None
            else ("FAIL" if selected is not None else "WARN"),
            "C++ compiler",
            compiler or f"{compiler_name} not found",
        )
    )
    if wmake is None and selected is not None:
        checks.append(("FAIL", "wmake", "not found in PATH"))
    else:
        checks.append(("PASS" if wmake else "WARN", "wmake", wmake or "not loaded"))

    mpi = shutil.which("mpirun") or shutil.which("mpiexec")
    checks.append(("PASS" if mpi else "WARN", "MPI launcher", mpi or "not detected"))

    if selected is None:
        checks.extend(
            (
                ("WARN", "Runtime profile", "load OpenFOAM to select an ABI"),
                ("WARN", "Longship", "runtime ABI not selected"),
                ("WARN", "OpenFOAM adapter", "runtime ABI not selected"),
                ("WARN", "ModelHost", "runtime ABI not selected"),
            )
        )
        return checks

    runtime = selected.runtime_dir
    runtime_profile = runtime / "runtime.yaml"
    checks.append(
        (
            "PASS" if runtime_profile.is_file() else "FAIL",
            "Runtime profile",
            str(runtime_profile),
        )
    )
    longship = runtime / "bin/foamnordic-longship"
    checks.append(
        (
            "PASS"
            if longship.is_file() and os.access(longship, os.X_OK)
            else "FAIL",
            "Longship",
            str(longship)
            if longship.is_file()
            else f"missing below {runtime / 'bin'}",
        )
    )
    adapter_candidates = tuple((runtime / "lib").glob("libfoamnordicOpenFOAM*"))
    checks.append(
        (
            "PASS" if adapter_candidates else "FAIL",
            "OpenFOAM adapter",
            str(adapter_candidates[0])
            if adapter_candidates
            else f"missing below {runtime / 'lib'}",
        )
    )
    model_host = runtime / "bin/foamnordic_model_worker"
    checks.append(
        (
            "PASS"
            if model_host.is_file() and os.access(model_host, os.X_OK)
            else "WARN",
            "ModelHost",
            str(model_host)
            if model_host.is_file()
            else "not installed (optional without ONNX)",
        )
    )
    return checks


def _doctor(stream: TextIO) -> int:
    import onsaemiro as osm

    checks = _doctor_checks()
    table = osm.TableMaker(
        title="FoamNordic Doctor",
        columns=["Status", "Check", "Detail"],
        mode="static",
    )
    for check in checks:
        table.add_row(*check)
    table.display()

    failures = sum(status == "FAIL" for status, _, _ in checks)
    warnings = sum(status == "WARN" for status, _, _ in checks)
    if failures:
        print(
            f"Result: unhealthy ({failures} failure(s), {warnings} warning(s))",
            file=stream,
        )
        return 1
    if warnings:
        print(f"Result: healthy with {warnings} warning(s)", file=stream)
    else:
        print("Result: healthy", file=stream)
    return 0


def _run_command(
    command: Sequence[str],
    log: TextIO,
    *,
    dry_run: bool,
    environment: dict[str, str] | None = None,
) -> None:
    if dry_run:
        print(f"$ {shlex.join(command)}", file=log)
        return
    subprocess.run(
        command,
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
    )


def _cmake_source(cache: Path) -> Path | None:
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
    value = next((line[len(prefix) :] for line in lines if line.startswith(prefix)), "")
    return Path(value).expanduser().resolve() if value else None


def _finalize_macos_openfoam_library(library_dir: Path, log: TextIO) -> Path:
    """Give each macOS adapter build a collision-proof install name."""

    library = library_dir / "libfoamnordicOpenFOAM.dylib"
    if not library.is_file():
        raise RuntimeError("wmake completed without installing the OpenFOAM library")
    identity = hashlib.sha256(library.read_bytes()).hexdigest()[:12]
    destination = library.with_name(f"libfoamnordicOpenFOAM-{identity}.dylib")
    install_name_tool = shutil.which("install_name_tool")
    if install_name_tool is None:
        raise RuntimeError("install_name_tool is required for the macOS OpenFOAM adapter")
    subprocess.run(
        [install_name_tool, "-id", f"@rpath/{destination.name}", str(library)],
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for stale in library_dir.glob("libfoamnordicOpenFOAM-*.dylib"):
        if stale != destination:
            stale.unlink()
    library.replace(destination)
    # Keep the generic linker name for ABI-matched tools built into the same
    # runtime. Runtime discovery and dictionary injection still select the
    # content-addressed library.
    library.symlink_to(destination.name)
    return destination


def _native_build_environment() -> dict[str, str]:
    """Prefer the compiler selected by the active native toolchain.

    Python environments such as Tykky/Conda may put compiler shims ahead of
    the compiler loaded with OpenFOAM.  OpenFOAM's wmake invokes ``gcc`` and
    ``g++`` by name, so put the directories containing CC/CXX first without
    removing the Python environment or tools installed in it.
    """
    environment = os.environ.copy()
    path = environment.get("PATH", "")
    compiler_directories: list[str] = []
    for variable in ("CXX", "CC"):
        command = environment.get(variable)
        if not command:
            continue
        try:
            executable = shlex.split(command)[0]
        except (IndexError, ValueError):
            continue
        resolved = shutil.which(executable, path=path)
        if resolved is not None:
            directory = str(Path(resolved).resolve().parent)
            if directory not in compiler_directories:
                compiler_directories.append(directory)

    if compiler_directories:
        existing = [entry for entry in path.split(os.pathsep) if entry]
        environment["PATH"] = os.pathsep.join(
            compiler_directories
            + [entry for entry in existing if entry not in compiler_directories]
        )
    return environment


def _build(args: argparse.Namespace, stream: TextIO) -> int:
    terminal = _Terminal(stream)
    terminal.section("FoamNordic native build")
    source = _source_root(args.source)

    if source is None:
        raise RuntimeError(
            "FoamNordic build kit is unavailable; reinstall FoamNordic from "
            "PyPI or GitHub, or pass a source checkout with --source"
        )

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake is required to build the native SDK")
    wmake = shutil.which("wmake")
    if wmake is None:
        raise RuntimeError(
            "wmake is unavailable; load OpenFOAM before running foamnordic build"
        )

    selected = profile(build_dir=args.build_dir, runtime_dir=args.prefix)
    assert selected is not None
    onnxruntime = (
        None if args.without_onnx else resolve_onnxruntime(download=not args.dry_run)
    )
    build_dir = selected.build_dir
    prefix = selected.runtime_dir
    log_path = build_dir / "foamnordic-build.log"
    configured_source = _cmake_source(build_dir / "CMakeCache.txt")
    if configured_source is not None and configured_source != source:
        if generated_kind(build_dir) != "build":
            raise RuntimeError(
                "the selected build directory belongs to another CMake source "
                f"and is not FoamNordic-owned: {build_dir}"
            )
        if args.dry_run:
            print(
                f"[FoamNordic] Would refresh build cache for source: {source}",
                file=stream,
            )
        else:
            shutil.rmtree(build_dir)
    build_was_present = build_dir.exists()
    prefix_was_present = prefix.exists()
    if not args.dry_run:
        build_dir.mkdir(parents=True, exist_ok=True)
        prefix.mkdir(parents=True, exist_ok=True)
        if not build_was_present:
            mark_generated(
                build_dir,
                kind="build",
                metadata={"platform": selected.platform, "openfoam": selected.openfoam},
            )
        if not prefix_was_present:
            mark_generated(
                prefix,
                kind="native-runtime",
                metadata={"platform": selected.platform, "openfoam": selected.openfoam},
            )

    adapter_source = build_dir / "openfoam"
    library_dir = prefix / "lib"
    application_dir = prefix / "bin"
    environment = _native_build_environment()
    environment.update(
        {
            "FOAMNORDIC_SOURCE": str(source),
            "FOAMNORDIC_BUILD": str(build_dir),
            "FOAM_USER_LIBBIN": str(library_dir),
            "FOAMNORDIC_OPENFOAM_LIB": str(library_dir),
            "FOAM_USER_APPBIN": str(application_dir),
        }
    )
    cmake_environment = environment.copy()
    if sys.platform == "darwin":
        # OpenFOAM.app ships its own libiconv.  Let Homebrew CMake resolve its
        # own dependencies, then restore the OpenFOAM environment for wmake.
        cmake_environment.pop("DYLD_LIBRARY_PATH", None)
        cmake_environment.pop("DYLD_FALLBACK_LIBRARY_PATH", None)
    configure = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFOAMNORDIC_TESTS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DFOAMNORDIC_ONNX_RUNTIME={'ON' if onnxruntime else 'OFF'}",
        f"-DFOAMNORDIC_RESIDENT_TOOLS={'ON' if onnxruntime else 'OFF'}",
    ]
    # Development installation always exports the inference core, including
    # Smedja and non-ONNX kernels. Build it even when the optional ONNX-backed
    # resident executable is disabled.
    runtime_targets = [
        "foamnordic_adapter",
        "foamnordic_inference",
        "foamnordic-longship",
    ]
    if onnxruntime is not None:
        configure.extend(
            (
                f"-DFOAMNORDIC_ONNX_RUNTIME_ROOT={onnxruntime.root}",
                f"-DFOAMNORDIC_ONNXRUNTIME_INCLUDE_DIR={onnxruntime.include}",
                f"-DFOAMNORDIC_ONNXRUNTIME_LIBRARY={onnxruntime.library}",
            )
        )
        runtime_targets.append("foamnordic_model_worker")
    commands: list[tuple[str, list[str]]] = [
        (
            "Configure native SDK",
            configure,
        ),
        (
            "Build native runtime",
            [
                cmake,
                "--build",
                str(build_dir),
                "--target",
                *runtime_targets,
                "--parallel",
                str(args.jobs),
            ],
        ),
        (
            "Install native C++ SDK",
            [
                cmake,
                "--install",
                str(build_dir),
                "--component",
                "Development",
            ],
        ),
    ]
    commands.append(
        (
            "Install native runtime tools",
            [cmake, "--install", str(build_dir), "--component", "Runtime"],
        )
    )
    commands.append(("Build OpenFOAM integration", [wmake, "libso"]))

    if args.dry_run:
        log: TextIO = stream
        for index, (description, command) in enumerate(commands, start=1):
            with terminal.step(index, len(commands), description):
                if description == "Build OpenFOAM integration":
                    print(f"$ cd {shlex.quote(str(adapter_source))}", file=log)
                _run_command(command, log, dry_run=True)
    else:
        try:
            if adapter_source.exists():
                shutil.rmtree(adapter_source)
            shutil.copytree(source / "src/foamnordic/openfoam", adapter_source)
            library_dir.mkdir(parents=True, exist_ok=True)
            application_dir.mkdir(parents=True, exist_ok=True)
            generic_macos_library = library_dir / "libfoamnordicOpenFOAM.dylib"
            if sys.platform == "darwin" and generic_macos_library.is_symlink():
                generic_macos_library.unlink()
            with log_path.open("w", encoding="utf-8") as log:
                for index, (description, command) in enumerate(commands, start=1):
                    with terminal.step(index, len(commands), description):
                        if description == "Build OpenFOAM integration":
                            subprocess.run(
                                command,
                                cwd=adapter_source,
                                env=environment,
                                check=True,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                            if sys.platform == "darwin":
                                _finalize_macos_openfoam_library(library_dir, log)
                        else:
                            _run_command(
                                command,
                                log,
                                dry_run=False,
                                environment=cmake_environment,
                            )
        except Exception:
            print(f"Build log: {log_path}", file=stream)
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-60:]), file=stream)
            raise

    print(f"\nOpenFOAM ABI:  {selected.openfoam}", file=stream)
    print(f"Native runtime: {prefix}", file=stream)
    if not args.dry_run:
        cache = build_dir / "CMakeCache.txt"
        integration = next(
            (
                path
                for path in (
                    *sorted(prefix.glob("lib/libfoamnordicOpenFOAM-*.dylib")),
                    prefix / "lib/libfoamnordicOpenFOAM.so",
                    prefix / "lib/libfoamnordicOpenFOAM.dylib",
                )
                if path.is_file()
            ),
            None,
        )
        if integration is None:
            raise RuntimeError("wmake completed without installing the OpenFOAM library")
        cache_text = (
            cache.read_text(encoding="utf-8", errors="ignore")
            if cache.is_file()
            else ""
        )
        if generated_kind(build_dir) is None and "FoamNordic" in cache_text:
            mark_generated(build_dir, kind="build")
        if generated_kind(prefix) is None and integration.is_file():
            mark_generated(
                prefix,
                kind="native-runtime",
                metadata={"platform": selected.platform, "openfoam": selected.openfoam},
            )
        print(f"Build log:  {log_path}", file=stream)
        if onnxruntime is not None:
            worker = prefix / "bin/foamnordic_model_worker"
            if not worker.is_file():
                raise RuntimeError(
                    "native build completed without installing ModelHost"
                )
            print(
                f"ONNX Runtime: {ONNXRUNTIME_VERSION} ({onnxruntime.source})",
                file=stream,
            )
            print(f"ModelHost: {worker}", file=stream)
        runtime_profile = write_runtime_profile(selected)
        print(f"Runtime profile: {runtime_profile}", file=stream)
    return 0


def _clobber_targets(args: argparse.Namespace) -> list[Path]:
    candidates = list(active_runtime_candidates())
    selected = profile(required=False)
    if selected is not None:
        candidates.append(selected.build_dir)
    for root in (
        Path.home() / ".cache/foamnordic/build",
        Path.home() / ".local/share/foamnordic/runtime",
    ):
        if root.is_dir():
            candidates.extend(
                marker.parent for marker in root.glob("*/*/.foamnordic-generated.json")
            )
    candidates.extend(
        (
            Path.home() / ".cache/foamnordic/build",
            Path.home() / ".local/share/foamnordic/native",
        )
    )
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


def _artifact_record(artifact) -> dict[str, object]:
    def ports(values) -> list[dict[str, object]]:
        return [
            {
                "name": value.name,
                "components": value.components,
                "dtype": value.dtype,
            }
            for value in values
        ]

    return {
        "path": str(artifact.path),
        "name": artifact.name,
        "container_version": artifact.container_version,
        "schema_version": artifact.schema_version,
        "backend": artifact.backend,
        "runtime": artifact.runtime,
        "bundled": artifact.bundled,
        "size": artifact.size,
        "payload_offset": artifact.payload_offset,
        "payload_size": artifact.payload_size,
        "inputs": ports(artifact.inputs),
        "outputs": ports(artifact.outputs),
        "input_scaler": artifact.input_scaler,
        "output_scaler": artifact.output_scaler,
    }


def _inspect_fnom(args: argparse.Namespace, stream: TextIO) -> int:
    from .models import load

    artifact = load(args.artifact)
    record = _artifact_record(artifact)
    if args.json:
        print(json.dumps(record, indent=2), file=stream)
        return 0

    terminal = _Terminal(stream)
    terminal.section(f"FNOM artifact: {artifact.name}")
    inputs = ", ".join(
        f"{item.name}[{item.components}] {item.dtype}" for item in artifact.inputs
    )
    outputs = ", ".join(
        f"{item.name}[{item.components}] {item.dtype}" for item in artifact.outputs
    )
    scalers = (
        f"input={artifact.input_scaler or 'none'}, "
        f"output={artifact.output_scaler or 'none'}"
    )
    payload = (
        f"embedded, {artifact.payload_size} B"
        if artifact.bundled
        else "external legacy payload"
    )
    values = (
        ("File", artifact.path),
        (
            "Container",
            f"FNOBND{artifact.container_version}"
            if artifact.bundled
            else "manifest-only",
        ),
        ("Manifest", f"schema {artifact.schema_version}"),
        ("Backend", artifact.backend),
        ("Runtime", artifact.runtime or "default"),
        ("Inputs", inputs),
        ("Outputs", outputs),
        ("Scalers", scalers),
        ("Payload", payload),
        ("File size", f"{artifact.size} B"),
    )
    width = max(len(name) for name, _ in values)
    for name, value in values:
        print(f"{name:<{width}}  {value}", file=stream)
    return 0


def _validate_fnom(args: argparse.Namespace, stream: TextIO) -> int:
    from .models import load

    artifact = load(args.artifact)
    container = (
        f"FNOBND{artifact.container_version}"
        if artifact.bundled
        else "manifest-only"
    )
    print(
        f"Valid FNOM: {artifact.path} "
        f"({container}, schema {artifact.schema_version}, "
        f"{artifact.backend})",
        file=stream,
    )
    return 0


def _compile_fnom(args: argparse.Namespace, stream: TextIO) -> int:
    from . import _native
    from .build.compiler import compile_source
    from .models import load

    artifact = load(args.artifact)
    if artifact.backend != "compiled" or artifact.runtime != "cpp-v1":
        raise ValueError("foamnordic compile requires a compiled cpp-v1 FNOM artifact")
    source = bytes(_native.read_model_payload(str(artifact.path)))
    library, cache_hit, seconds = compile_source(source)
    action = "Reused" if cache_hit else "Compiled"
    print(f"{action} FNOM: {artifact.path}", file=stream)
    print(f"Native library: {library}", file=stream)
    if not cache_hit:
        print(f"Compile time: {seconds:.3f} s", file=stream)
    return 0


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="foamnordic",
        description="Build and run native OpenFOAM closure workloads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(
        dest="command",
        metavar="[command]",
        required=True,
    )
    directories = subcommands.add_parser(
        "dir",
        help="show the active package, environment, and native directories",
        description="Show the directories used by the active FoamNordic installation.",
    )
    directory_paths = directories.add_mutually_exclusive_group()
    directory_paths.add_argument(
        "--runtime", action="store_true", help="print only the active runtime prefix"
    )
    directory_paths.add_argument(
        "--include", action="store_true", help="print only the installed SDK include path"
    )
    directory_paths.add_argument(
        "--openfoam-library",
        action="store_true",
        help="print only the exact ABI-matched OpenFOAM integration library",
    )
    subcommands.add_parser(
        "doctor",
        help="diagnose the active Python, native, and OpenFOAM environment",
        description=(
            "Run fast, read-only checks for the active FoamNordic installation, "
            "native extension, OpenFOAM toolchain, MPI, and ABI-matched runtime."
        ),
    )
    build = subcommands.add_parser(
        "build",
        help="build the runtime for the active OpenFOAM ABI",
        description=(
            "Build the native C++ core and OpenFOAM integration for the currently "
            "loaded OpenFOAM ABI, then install them in the user runtime directory."
        ),
    )
    build.add_argument("--source", type=Path, help="FoamNordic source tree")
    build.add_argument("--build-dir", type=Path, help="CMake build directory")
    build.add_argument("--prefix", type=Path, help="native runtime install prefix")
    build.add_argument("--jobs", type=int, default=_default_jobs(), help="parallel jobs")
    build.add_argument(
        "--without-onnx",
        action="store_true",
        help="skip the native ONNX ModelHost",
    )
    build.add_argument("--dry-run", action="store_true", help="show commands only")
    inspect_artifact = subcommands.add_parser(
        "inspect",
        help="show the execution contract of an FNOM artifact",
        description=(
            "Read and validate an FNOM container and display its model backend, "
            "tensor contract, scalers, runtime, and payload layout."
        ),
    )
    inspect_artifact.add_argument("artifact", type=Path, metavar="MODEL.fnom")
    inspect_artifact.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable metadata",
    )
    validate_artifact = subcommands.add_parser(
        "validate",
        help="validate an FNOM container and execution contract",
        description=(
            "Validate FNOM structure, bounds, versions, tensor contracts, scalers, "
            "and embedded payload layout without executing the model payload."
        ),
    )
    validate_artifact.add_argument("artifact", type=Path, metavar="MODEL.fnom")
    compile_artifact = subcommands.add_parser(
        "compile",
        help="warm the target-native cache for a compiled FNOM artifact",
        description=(
            "Compile a portable cpp-v1 FNOM payload for this target and retain "
            "the shared library in FoamNordic's content-addressed cache."
        ),
    )
    compile_artifact.add_argument("artifact", type=Path, metavar="MODEL.fnom")
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
        try:
            return _directories(args, sys.stdout)
        except RuntimeError as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    if args.command == "doctor":
        return _doctor(sys.stdout)
    if args.command == "build":
        if args.jobs <= 0:
            parser.error("--jobs must be positive")
        try:
            return _build(args, sys.stdout)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    if args.command == "inspect":
        try:
            return _inspect_fnom(args, sys.stdout)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    if args.command == "validate":
        try:
            return _validate_fnom(args, sys.stdout)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"foamnordic: error: {error}", file=sys.stderr)
            return 1
    if args.command == "compile":
        try:
            return _compile_fnom(args, sys.stdout)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
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
