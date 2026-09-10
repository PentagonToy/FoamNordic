"""Publish complete runtime generations without rewriting an active runtime."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def fingerprint(source: Path, options: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    paths = [source / "CMakeLists.txt"]
    for directory in (
        "src",
        "tools",
        "packages/bindnordic/src/bindnordic/native",
        "packages/foamnordic/buildkit",
        "packages/foamnordic/foamnordic/build",
    ):
        root = source / directory
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file()
                         and "lnInclude" not in p.parts
                         and not any(part.startswith(("linux64", "darwin64", "__pycache__")) for part in p.parts))
    paths.append(source / "packages/foamnordic/foamnordic/_cli.py")
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"build-receipt.json", ".generation.lock"}
        and ".pins" not in path.parts
    }


def complete(root: Path, identity: str) -> bool:
    try:
        receipt = json.loads((root / "build-receipt.json").read_text())
        return receipt["identity"] == identity and receipt["files"] == inventory(root)
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def publish(root: Path, generation: Path, identity: str) -> None:
    (generation / "build-receipt.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "files": inventory(generation),
                "published_at_ns": time.time_ns(),
            },
            sort_keys=True,
        )
    )
    link = root / (".current-" + uuid.uuid4().hex)
    try:
        link.symlink_to(generation.relative_to(root), target_is_directory=True)
        os.replace(link, root / "current")
    finally:
        link.unlink(missing_ok=True)


def pin(generation: Path, owner: str) -> Path:
    """Pin one published generation until the returned marker is removed."""

    selected = generation.resolve()
    if selected.parent.name != "generations":
        raise ValueError(f"not a FoamNordic runtime generation: {selected}")
    pins = selected / ".pins"
    pins.mkdir(mode=0o700, exist_ok=True)
    marker = pins / uuid.uuid4().hex
    marker.write_text(owner + "\n", encoding="utf-8")
    return marker


def collect_garbage(root: Path, *, keep: int = 2) -> tuple[Path, ...]:
    """Remove old complete, unpinned generations while preserving current."""

    if keep < 1:
        raise ValueError("generation retention must be positive")
    generations = root / "generations"
    if not generations.is_dir():
        return ()
    current = (root / "current").resolve() if (root / "current").is_dir() else None
    def published(path: Path) -> int:
        try:
            return int(json.loads((path / "build-receipt.json").read_text()).get(
                "published_at_ns", path.stat().st_mtime_ns
            ))
        except (OSError, ValueError, TypeError):
            return path.stat().st_mtime_ns

    candidates = sorted(
        (path for path in generations.iterdir() if path.is_dir()),
        key=published,
        reverse=True,
    )
    preserved = set(candidates[:keep])
    if current is not None:
        preserved.add(current)
    removed = []
    for candidate in candidates:
        if candidate in preserved or any((candidate / ".pins").glob("*")):
            continue
        try:
            receipt = json.loads((candidate / "build-receipt.json").read_text())
            identity = str(receipt["identity"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not complete(candidate, identity):
            continue
        shutil.rmtree(candidate)
        removed.append(candidate)
    return tuple(removed)


def tool_identity(name: str) -> dict[str, object]:
    path = shutil.which(name)
    if not path:
        return {"name": name, "path": None}
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"name": name, "path": str(resolved), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def source_identity(root: Path) -> dict[str, object]:
    """Hash a small external source tree used by the native build."""

    selected = root.resolve()
    digest = hashlib.sha256()
    files = sorted(path for path in selected.rglob("*") if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(selected)).encode())
        digest.update(path.read_bytes())
    return {"files": len(files), "sha256": digest.hexdigest()}


def dependency_identity(environment: dict[str, str], onnx) -> dict[str, object]:
    """Track external headers, libraries and OpenFOAM build rules in place."""
    paths = []
    project = environment.get("WM_PROJECT_DIR")
    if project:
        for directory in ("src", "wmake"):
            root = Path(project) / directory
            paths.extend(p for p in root.rglob("*") if p.is_file()
                         and "lnInclude" not in p.parts
                         and not any(part.startswith(("linux64", "darwin64")) for part in p.parts))
    library = environment.get("FOAM_LIBBIN")
    if library:
        paths.extend(p for p in Path(library).glob("*") if p.is_file())
    if onnx is not None:
        paths.extend(p for p in onnx.include.rglob("*") if p.is_file())
        paths.append(onnx.library)
    digest = hashlib.sha256()
    # Dependency timestamps detect in-place site upgrades without reading several
    # gigabytes of OpenFOAM libraries on every invocation.
    for path in sorted(set(paths)):
        stat = path.stat()
        digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode())
    return {"files": len(paths), "metadata_sha256": digest.hexdigest()}
