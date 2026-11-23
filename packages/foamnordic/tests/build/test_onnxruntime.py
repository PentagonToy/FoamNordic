from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock, call
from urllib.error import HTTPError, URLError
import zipfile

import pytest
from foamnordic.build import onnxruntime
from foamnordic.build.onnxruntime import resolve


def test_explicit_root_requires_api_28(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        include = root / "include"
        library = root / "lib"
        include.mkdir()
        library.mkdir()
        (include / "onnxruntime_cxx_api.h").touch()
        (include / "onnxruntime_c_api.h").write_text(
            "#define ORT_API_VERSION 28\n", encoding="utf-8"
        )
        library_name = (
            "libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so"
        )
        (library / library_name).touch()
        monkeypatch.setenv("FOAMNORDIC_ONNX_RUNTIME_ROOT", str(root))
        selected = resolve(download=False)
        assert selected.root == root.resolve()
        assert selected.source == "environment"

        (include / "onnxruntime_c_api.h").write_text(
            "#define ORT_API_VERSION 29\n", encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="API 28"):
            resolve(download=False)


def _sdk_archive(*, nuget=True, api=28, extra=None):
    members = {
        "include/onnxruntime_cxx_api.h": b"// C++ API\n",
        "include/onnxruntime_c_api.h": f"#define ORT_API_VERSION {api}\n".encode(),
        "lib/libonnxruntime.so": b"linux runtime",
        "lib/libonnxruntime.dylib": b"macOS runtime",
    }
    archive = io.BytesIO()
    if nuget:
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("LICENSE", b"MIT License")
            bundle.writestr("ThirdPartyNotices.txt", b"Third-party notices")
            for name, content in members.items():
                if name.startswith("include/"):
                    bundle.writestr(f"build/native/{name}", content)
                else:
                    for rid in ("linux-x64", "linux-arm64", "osx-arm64"):
                        bundle.writestr(
                            f"runtimes/{rid}/native/{Path(name).name}",
                            content + rid.encode(),
                        )
            if extra is not None:
                bundle.writestr(extra, b"unsafe")
    else:
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for name, content in members.items():
                member = tarfile.TarInfo(f"onnxruntime/{name}")
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))

    return archive.getvalue()


def test_github_success_does_not_request_nuget(tmp_path, monkeypatch):
    payload = _sdk_archive(nuget=False)
    request = Mock(return_value=io.BytesIO(payload))
    monkeypatch.setattr(onnxruntime, "urlopen", request)
    root = tmp_path / "sdk"
    onnxruntime._download(root, "sdk.tgz", hashlib.sha256(payload).hexdigest())

    assert onnxruntime._inspect(root, "download") is not None
    request.assert_called_once_with(f"{onnxruntime._BASE_URL}/sdk.tgz", timeout=60)


@pytest.mark.parametrize(
    "key,rid",
    [
        (("linux", "x86_64"), "linux-x64"),
        (("linux", "aarch64"), "linux-arm64"),
        (("darwin", "arm64"), "osx-arm64"),
    ],
)
@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("github", 404, "Not Found", None, None),
        URLError("offline"),
        TimeoutError("timed out"),
    ],
)
def test_network_failure_uses_nuget(tmp_path, monkeypatch, caplog, key, rid, failure):
    payload = _sdk_archive()
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: key)
    monkeypatch.setattr(
        onnxruntime.platform, "system",
        lambda: "Darwin" if key[0] == "darwin" else "Linux",
    )
    monkeypatch.setattr(onnxruntime.platform, "mac_ver", lambda: ("14.0", (), "arm64"))
    monkeypatch.setattr(onnxruntime, "_NUGET_SHA256", hashlib.sha256(payload).hexdigest())
    request = Mock(side_effect=[failure, io.BytesIO(payload)])
    monkeypatch.setattr(onnxruntime, "urlopen", request)
    root = tmp_path / "sdk"
    onnxruntime._download(root, "sdk.tgz", "unused")

    selected = onnxruntime._inspect(root, "download")
    assert selected is not None
    assert selected.library.read_bytes().endswith(rid.encode())
    assert (root / "LICENSE").read_bytes() == b"MIT License"
    assert (root / "ThirdPartyNotices.txt").is_file()
    alias = (
        "libonnxruntime.1.dylib" if key[0] == "darwin"
        else "libonnxruntime.so.1"
    )
    assert (root / "lib" / alias).resolve() == selected.library
    assert request.call_args_list == [
        call(f"{onnxruntime._BASE_URL}/sdk.tgz", timeout=60),
        call(onnxruntime._NUGET_URL, timeout=60),
    ]
    assert "trying official NuGet" in caplog.text


def test_both_downloads_fail_with_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(onnxruntime, "urlopen", Mock(side_effect=URLError("offline")))

    with pytest.raises(RuntimeError, match="FOAMNORDIC_ONNX_RUNTIME_ROOT") as error:
        onnxruntime._download(tmp_path / "sdk", "sdk.tgz", "unused")

    assert onnxruntime._BASE_URL in str(error.value)
    assert onnxruntime._NUGET_URL in str(error.value)
    assert "--without-onnx" in str(error.value)
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("nuget", [False, True])
def test_checksum_failure_preserves_cache(tmp_path, monkeypatch, nuget):
    request = Mock()
    monkeypatch.setattr(onnxruntime, "urlopen", request)
    request.side_effect = (
        [URLError("offline"), io.BytesIO(b"corrupt")]
        if nuget else [io.BytesIO(b"corrupt")]
    )
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("linux", "x86_64"))
    root = tmp_path / "sdk"
    root.mkdir()
    (root / "keep").write_text("existing cache")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        onnxruntime._download(root, "sdk.tgz", "unused")

    assert request.call_count == (2 if nuget else 1)
    assert (root / "keep").read_text() == "existing cache"


@pytest.mark.parametrize(
    "member",
    [
        "build/native/include/../../../../escape",
        "runtimes/linux-x64/native/../../../../escape",
    ],
)
def test_nuget_rejects_path_traversal(tmp_path, monkeypatch, member):
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("linux", "x86_64"))
    archive = tmp_path / "sdk.nupkg"
    archive.write_bytes(_sdk_archive(extra=member))

    with pytest.raises(RuntimeError, match="unsafe path"):
        onnxruntime._unpack_nuget(archive, tmp_path / "sdk")

    assert not (tmp_path / "escape").exists()


def test_nuget_rejects_incompatible_api(tmp_path, monkeypatch):
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("linux", "x86_64"))
    archive = tmp_path / "sdk.nupkg"
    archive.write_bytes(_sdk_archive(api=29))

    with pytest.raises(RuntimeError, match="incomplete"):
        onnxruntime._unpack_nuget(archive, tmp_path / "sdk")


def test_nuget_requires_supported_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("darwin", "arm64"))
    monkeypatch.setattr(onnxruntime.platform, "mac_ver", lambda: ("13.7", (), "arm64"))
    request = Mock(side_effect=URLError("offline"))
    monkeypatch.setattr(onnxruntime, "urlopen", request)

    with pytest.raises(RuntimeError, match="requires macOS 14"):
        onnxruntime._download(tmp_path / "sdk", "sdk.tgz", "unused")

    assert request.call_count == 1


def test_dry_run_does_not_download(tmp_path, monkeypatch):
    monkeypatch.delenv("FOAMNORDIC_ONNX_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr(onnxruntime, "_platform_key", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(onnxruntime, "_inspect", Mock(return_value=None))
    monkeypatch.setattr(onnxruntime, "_brew_root", lambda: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    request = Mock()
    monkeypatch.setattr(onnxruntime, "urlopen", request)

    selected = resolve(download=False)

    assert selected.source == "download"
    assert not selected.root.exists()
    request.assert_not_called()
