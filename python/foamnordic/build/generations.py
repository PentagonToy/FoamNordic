"""Publish complete runtime generations without rewriting an active runtime."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def fingerprint(source: Path, options: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    paths = [source / "CMakeLists.txt"]
    for directory in ("src", "tools", "python/buildkit", "python/foamnordic/build"):
        root = source / directory
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file()
                         and "lnInclude" not in p.parts
                         and not any(part.startswith(("linux64", "darwin64", "__pycache__")) for part in p.parts))
    paths.append(source / "python/foamnordic/_cli.py")
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "build-receipt.json"
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
        json.dumps({"identity": identity, "files": inventory(generation)}, sort_keys=True)
    )
    link = root / (".current-" + uuid.uuid4().hex)
    try:
        link.symlink_to(generation.relative_to(root), target_is_directory=True)
        os.replace(link, root / "current")
    finally:
        link.unlink(missing_ok=True)


def tool_identity(name: str) -> dict[str, object]:
    path = shutil.which(name)
    if not path:
        return {"name": name, "path": None}
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"name": name, "path": str(resolved), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


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
