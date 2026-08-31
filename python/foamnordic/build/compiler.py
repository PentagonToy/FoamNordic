"""Persistent target-native compilation cache for compiled FNOM payloads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
from time import monotonic, sleep


def _compiler() -> str:
    explicit = os.environ.get("CXX")
    candidates = (explicit,) if explicit else ("c++", "clang++", "g++")
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError("compiled FNOM requires a C++17 compiler (CXX, c++, clang++, or g++)")


def _identity(compiler: str) -> str:
    result = subprocess.run(
        [compiler, "--version"], check=True, text=True, capture_output=True
    )
    first_line = result.stdout.splitlines()[0] if result.stdout else compiler
    return f"{platform.system()}|{platform.machine()}|{first_line}|cpp-v1|-O3|-pthread"


def compile_source(source: bytes) -> tuple[Path, bool, float]:
    """Return cached library path, cache-hit state, and compilation seconds."""

    compiler = _compiler()
    digest = hashlib.sha256(_identity(compiler).encode() + b"\0" + source).hexdigest()
    root = Path(
        os.environ.get(
            "FOAMNORDIC_COMPILED_CACHE",
            Path.home() / ".cache" / "foamnordic" / "compiled",
        )
    ).expanduser()
    destination = root / digest
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = destination / f"model{suffix}"
    if library.is_file():
        return library, True, 0.0
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".compile.lock"
    acquired = False
    deadline = monotonic() + 300.0
    while monotonic() < deadline:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            acquired = True
            break
        except FileExistsError:
            if library.is_file():
                return library, True, 0.0
            sleep(0.05)
    if not acquired:
        raise TimeoutError(f"timed out waiting for compiled FNOM cache: {destination}")
    started = monotonic()
    try:
        if library.is_file():
            return library, True, 0.0
        with tempfile.TemporaryDirectory(prefix="foamnordic-compile-", dir=destination) as tmp:
            temporary = Path(tmp)
            source_path = temporary / "model.cpp"
            output_path = temporary / f"model{suffix}"
            source_path.write_bytes(source)
            command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-pthread"]
            if platform.system() == "Darwin":
                command.extend(("-dynamiclib", str(source_path), "-o", str(output_path)))
            else:
                command.extend(("-shared", "-fPIC", str(source_path), "-o", str(output_path)))
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                rendered = shlex.join(command)
                raise RuntimeError(
                    f"compiled FNOM build failed: {rendered}\n{result.stderr.strip()}"
                )
            os.replace(output_path, library)
        return library, False, monotonic() - started
    finally:
        lock.unlink(missing_ok=True)


__all__ = ["compile_source"]
