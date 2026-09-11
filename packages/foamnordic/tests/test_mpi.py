from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml
from foamnordic.execution.mpi import discover_mpi_policy, write_runtime_profile
from foamnordic.execution.runtime_paths import RuntimeProfile


def test_macos_wrapper_discovers_and_isolates_homebrew_mpi(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        launcher = Path(directory) / "mpirun"
        launcher.touch(mode=0o755)
        _mock_patch_1 = mocker.patch.dict(os.environ, {}, clear=True)
        _mock_patch_2 = mocker.patch(
            "foamnordic.execution.mpi.platform.system", return_value="Darwin"
        )
        _mock_patch_3 = mocker.patch(
            "foamnordic.execution.mpi._brew_mpirun", return_value=launcher
        )
        try:
            policy = discover_mpi_policy(wrapper=True)
        finally:
            mocker.stop(_mock_patch_3)
            mocker.stop(_mock_patch_2)
            mocker.stop(_mock_patch_1)

    assert policy.launcher == launcher
    assert policy.isolate
    assert policy.source == "macos-auto"


def test_linux_keeps_inherited_mpi_policy(mocker) -> None:
    _mock_patch_4 = mocker.patch.dict(os.environ, {}, clear=True)
    _mock_patch_5 = mocker.patch(
        "foamnordic.execution.mpi.platform.system", return_value="Linux"
    )
    try:
        policy = discover_mpi_policy(wrapper=False)
    finally:
        mocker.stop(_mock_patch_5)
        mocker.stop(_mock_patch_4)

    assert policy.launcher is None
    assert not policy.isolate


def test_runtime_profile_round_trip(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        launcher = root / "mpirun"
        launcher.touch(mode=0o755)
        runtime = RuntimeProfile(
            platform="darwin-aarch64",
            openfoam="openfoam-v2606-darwin64ClangDPInt32Opt",
            build_dir=root / "build",
            runtime_dir=root,
        )
        _mock_patch_6 = mocker.patch.dict(
            os.environ, {"FOAMNORDIC_MPIRUN": str(launcher)}, clear=True
        )
        _mock_patch_7 = mocker.patch(
            "foamnordic.execution.mpi.platform.system", return_value="Darwin"
        )
        _mock_patch_8 = mocker.patch(
            "foamnordic.execution.mpi.platform.machine", return_value="arm64"
        )
        try:
            path = write_runtime_profile(runtime)
        finally:
            mocker.stop(_mock_patch_8)
            mocker.stop(_mock_patch_7)
            mocker.stop(_mock_patch_6)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        _mock_patch_9 = mocker.patch.dict(os.environ, {}, clear=True)
        _mock_patch_10 = mocker.patch(
            "foamnordic.execution.mpi.platform.system", return_value="Darwin"
        )
        try:
            policy = discover_mpi_policy(wrapper=True, runtime_dir=root)
        finally:
            mocker.stop(_mock_patch_10)
            mocker.stop(_mock_patch_9)

    assert document["platform"]["tag"] == "darwin-aarch64"
    assert document["mpi"]["mode"] == "external-isolated"
    assert policy.launcher == launcher.resolve()
    assert policy.isolate
    assert policy.source == "runtime-profile"
