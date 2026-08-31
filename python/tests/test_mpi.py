from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from foamnordic.execution.mpi import discover_mpi_policy, write_runtime_profile
from foamnordic.execution.runtime_paths import RuntimeProfile


class MpiPolicyTests(unittest.TestCase):
    def test_macos_wrapper_discovers_and_isolates_homebrew_mpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "mpirun"
            launcher.touch(mode=0o755)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("foamnordic.execution.mpi.platform.system", return_value="Darwin"),
                mock.patch("foamnordic.execution.mpi._brew_mpirun", return_value=launcher),
            ):
                policy = discover_mpi_policy(wrapper=True)

        self.assertEqual(policy.launcher, launcher)
        self.assertTrue(policy.isolate)
        self.assertEqual(policy.source, "macos-auto")

    def test_linux_keeps_inherited_mpi_policy(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("foamnordic.execution.mpi.platform.system", return_value="Linux"),
        ):
            policy = discover_mpi_policy(wrapper=False)

        self.assertIsNone(policy.launcher)
        self.assertFalse(policy.isolate)

    def test_runtime_profile_round_trip(self) -> None:
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
            with (
                mock.patch.dict(
                    os.environ, {"FOAMNORDIC_MPIRUN": str(launcher)}, clear=True
                ),
                mock.patch("foamnordic.execution.mpi.platform.system", return_value="Darwin"),
                mock.patch("foamnordic.execution.mpi.platform.machine", return_value="arm64"),
            ):
                path = write_runtime_profile(runtime)
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("foamnordic.execution.mpi.platform.system", return_value="Darwin"),
            ):
                policy = discover_mpi_policy(wrapper=True, runtime_dir=root)

        self.assertEqual(document["platform"]["tag"], "darwin-aarch64")
        self.assertEqual(document["mpi"]["mode"], "external-isolated")
        self.assertEqual(policy.launcher, launcher.resolve())
        self.assertTrue(policy.isolate)
        self.assertEqual(policy.source, "runtime-profile")


if __name__ == "__main__":
    unittest.main()
