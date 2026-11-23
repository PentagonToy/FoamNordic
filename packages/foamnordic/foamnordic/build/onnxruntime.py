"""Resolve the pinned native ONNX Runtime used by ModelHost."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import tarfile
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import zipfile


VERSION = "1.28.0"
API_VERSION = 28
_BASE_URL = f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}"
_NUGET_URL = (
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/"
    f"{VERSION}/microsoft.ml.onnxruntime.{VERSION}.nupkg"
)
_NUGET_SHA256 = "769d1d3ea8ab6cd69f737c9dd4d4462aa4ad0ccfa106eaf506efc40d7bead5db"
_NUGET_RIDS = {
    ("darwin", "arm64"): "osx-arm64",
    ("linux", "x86_64"): "linux-x64",
    ("linux", "aarch64"): "linux-arm64",
}
_ASSETS = {
    ("darwin", "arm64"): (
        f"onnxruntime-osx-arm64-{VERSION}.tgz",
        "1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8",
    ),
    ("linux", "x86_64"): (
        f"onnxruntime-linux-x64-{VERSION}.tgz",
        "a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407",
    ),
    ("linux", "aarch64"): (
        f"onnxruntime-linux-aarch64-{VERSION}.tgz",
        "e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb",
    ),
}


@dataclass(frozen=True, slots=True)
class NativeOnnxRuntime:
    root: Path
    include: Path
    library: Path
    source: str


def _platform_key() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    architecture = {"amd64": "x86_64", "arm64": "arm64"}.get(machine, machine)
    if system == "linux" and architecture == "arm64":
        architecture = "aarch64"
    return system, architecture


def _inspect(root: Path, source: str) -> NativeOnnxRuntime | None:
    root = root.expanduser().resolve()
    headers = (
        root / "include/onnxruntime_cxx_api.h",
        root / "include/onnxruntime/onnxruntime_cxx_api.h",
    )
    header = next((path for path in headers if path.is_file()), None)
    if header is None:
        return None
    c_header = header.with_name("onnxruntime_c_api.h")
    try:
        match = re.search(
            r"^#define\s+ORT_API_VERSION\s+(\d+)",
            c_header.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    except OSError:
        return None
    if match is None or int(match.group(1)) != API_VERSION:
        return None
    names = (
        ("libonnxruntime.dylib", "libonnxruntime.1.dylib")
        if platform.system() == "Darwin"
        else ("libonnxruntime.so", f"libonnxruntime.so.{VERSION}")
    )
    library = next((root / "lib" / name for name in names if (root / "lib" / name).is_file()), None)
    if library is None:
        return None
    return NativeOnnxRuntime(root, header.parent, library, source)


def _brew_root() -> Path | None:
    brew = shutil.which("brew")
    if brew is None:
        return None
    import subprocess

    completed = subprocess.run(
        [brew, "--prefix", "onnxruntime"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip())


def _fetch(url: str, archive: Path, checksum: str) -> None:
    digest = hashlib.sha256()
    with urlopen(url, timeout=60) as response, archive.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)

    if digest.hexdigest() != checksum:
        raise RuntimeError(f"ONNX Runtime {VERSION} archive checksum mismatch: {url}")


def _unpack_nuget(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    system, architecture = _platform_key()
    rid = _NUGET_RIDS[(system, architecture)]
    prefixes = {
        "build/native/include/": destination / "include",
        f"runtimes/{rid}/native/": destination / "lib",
    }
    with zipfile.ZipFile(archive) as bundle:
        for name in ("LICENSE", "ThirdPartyNotices.txt"):
            (destination / name).write_bytes(bundle.read(name))
        for member in bundle.infolist():
            if member.is_dir():
                continue
            for prefix, directory in prefixes.items():
                if not member.filename.startswith(prefix):
                    continue
                target = (directory / member.filename[len(prefix):]).resolve()
                if directory.resolve() not in target.parents:
                    raise RuntimeError("ONNX Runtime archive contains an unsafe path")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                break

    if _inspect(destination, "download") is None:
        raise RuntimeError(f"NuGet ONNX Runtime {VERSION} is incomplete for {rid}")

    # NuGet ships unversioned filenames; native consumers use the library ID
    library = destination / "lib"
    if system == "darwin":
        (library / "libonnxruntime.1.dylib").symlink_to("libonnxruntime.dylib")
    else:
        (library / "libonnxruntime.so.1").symlink_to("libonnxruntime.so")


def _download(root: Path, asset: str, checksum: str) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as directory:
        stage = Path(directory)
        archive = stage / asset
        github_url = f"{_BASE_URL}/{asset}"
        nuget = False
        try:
            _fetch(github_url, archive, checksum)
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as github_error:
            if (
                _platform_key()[0] == "darwin"
                and int(platform.mac_ver()[0].split(".")[0]) < 14
            ):
                raise RuntimeError(
                    f"ONNX Runtime download failed: {github_url}: {github_error}. "
                    "The NuGet fallback requires macOS 14 or later; set "
                    "FOAMNORDIC_ONNX_RUNTIME_ROOT to a compatible SDK or use --without-onnx."
                ) from github_error
            logging.getLogger(__name__).warning(
                "ONNX Runtime download failed: %s: %s; trying official NuGet package %s",
                github_url,
                github_error,
                _NUGET_URL,
            )
            archive = stage / "onnxruntime.nupkg"
            try:
                _fetch(_NUGET_URL, archive, _NUGET_SHA256)
            except (URLError, TimeoutError, ConnectionError, IncompleteRead) as nuget_error:
                raise RuntimeError(
                    f"ONNX Runtime {VERSION} downloads failed. "
                    f"GitHub: {github_url}: {github_error}. "
                    f"NuGet: {_NUGET_URL}: {nuget_error}. "
                    "Set FOAMNORDIC_ONNX_RUNTIME_ROOT to an unpacked SDK "
                    "or use --without-onnx."
                ) from nuget_error
            nuget = True

        unpacked = stage / "unpacked"
        unpacked.mkdir()
        if nuget:
            _unpack_nuget(archive, unpacked)
            source = unpacked
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    destination = (unpacked / member.name).resolve()
                    if unpacked.resolve() not in destination.parents and destination != unpacked.resolve():
                        raise RuntimeError("ONNX Runtime archive contains an unsafe path")
                bundle.extractall(unpacked, filter="data")
            children = tuple(unpacked.iterdir())
            source = children[0] if len(children) == 1 and children[0].is_dir() else unpacked

        if _inspect(source, "download") is None:
            raise RuntimeError(f"downloaded ONNX Runtime {VERSION} is incomplete")
        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(source), str(root))


def resolve(*, download: bool = True) -> NativeOnnxRuntime:
    """Find API-28 ONNX Runtime or cache the official platform archive."""

    explicit = os.environ.get("FOAMNORDIC_ONNX_RUNTIME_ROOT")
    if explicit:
        selected = _inspect(Path(explicit), "environment")
        if selected is None:
            raise RuntimeError(
                "FOAMNORDIC_ONNX_RUNTIME_ROOT must contain ONNX Runtime "
                f"{VERSION} (API {API_VERSION}) headers and library"
            )
        return selected
    candidates: list[tuple[Path, str]] = []
    if platform.system() == "Darwin":
        brew = _brew_root()
        if brew is not None:
            candidates.append((brew, "homebrew"))
    candidates.extend((Path(path), "system") for path in ("/usr/local", "/usr"))
    for root, source in candidates:
        selected = _inspect(root, source)
        if selected is not None:
            return selected
    key = _platform_key()
    try:
        asset, checksum = _ASSETS[key]
    except KeyError:
        raise RuntimeError(
            "native ONNX ModelHost supports linux-x86_64, linux-aarch64, "
            "and macOS arm64; use --without-onnx for another platform"
        ) from None
    root = (
        Path.home()
        / ".cache/foamnordic/dependencies/onnxruntime"
        / VERSION
        / f"{key[0]}-{key[1]}"
    )
    selected = _inspect(root, "cache")
    if selected is not None:
        return selected
    if not download:
        return NativeOnnxRuntime(
            root,
            root / "include",
            root / "lib/libonnxruntime.dylib"
            if key[0] == "darwin"
            else root / "lib/libonnxruntime.so",
            "download",
        )
    _download(root, asset, checksum)
    selected = _inspect(root, "download")
    if selected is None:
        raise RuntimeError(f"downloaded ONNX Runtime {VERSION} is incomplete")
    return selected
