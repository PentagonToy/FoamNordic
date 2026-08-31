from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import foamnordic as fno
from foamnordic._cli import _source_root, main
from foamnordic.build.onnxruntime import NativeOnnxRuntime
from foamnordic.core.managed import MARKER, mark_generated
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.runtime_paths import openfoam_abi_for_toolchain


class CliTests(unittest.TestCase):
    def test_case_toolchain_probe_discovers_openfoam_abi(self) -> None:
        completed = subprocess.CompletedProcess(
            args=("bash",),
            returncode=0,
            stdout=(
                "module noise\n"
                "__FOAMNORDIC_VERSION__=v2512\n"
                "__FOAMNORDIC_OPTIONS__=linux64GccDPInt32-spack\n"
            ),
            stderr="",
        )
        toolchain = SimpleNamespace(
            command="module load openfoam/2512",
            wrapper=False,
            shell="bash",
        )
        with patch("foamnordic.execution.runtime_paths.subprocess.run", return_value=completed):
            self.assertEqual(
                openfoam_abi_for_toolchain(toolchain),
                "openfoam-v2512-linux64GccDPInt32-spack",
            )

    def test_build_discovers_kit_bundled_beside_installed_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "site-packages/foamnordic"
            buildkit = package / "buildkit"
            (buildkit / "src/foamnordic/runtime").mkdir(parents=True)
            (buildkit / "CMakeLists.txt").touch()
            unrelated = root / "working-directory"
            unrelated.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(unrelated)
                with patch(
                    "foamnordic._cli.__file__",
                    str(package / "_cli.py"),
                ):
                    self.assertEqual(_source_root(None), buildkit.resolve())
            finally:
                os.chdir(previous)

    def test_public_version_and_directory_are_discoverable(self) -> None:
        self.assertEqual(fno.__version__, "1.0.4")
        self.assertIn("Longship", dir(fno))
        self.assertIn("export", dir(fno))

    def test_cli_version_is_discoverable(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as stopped, redirect_stdout(output):
            main(["--version"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "foamnordic 1.0.4")

    def test_top_level_help_lists_dir_build_and_clobber(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as stopped, redirect_stdout(output):
            main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("dir", output.getvalue())
        self.assertIn("build", output.getvalue())
        self.assertIn("clobber", output.getvalue())

    def test_dir_reports_active_installation_paths(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = main(["dir"])
        self.assertEqual(status, 0)
        report = output.getvalue()
        self.assertIn("FoamNordic directories", report)
        self.assertIn("Python environment", report)
        self.assertIn("Python package", report)
        self.assertIn("Native module", report)

    def test_build_help_is_discoverable(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as stopped, redirect_stdout(output):
            main(["build", "--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--source", output.getvalue())
        self.assertIn("--dry-run", output.getvalue())
        self.assertIn("--without-onnx", output.getvalue())

    def test_source_build_dry_run_is_visual_and_non_mutating(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            prefix = root / "install"
            onnxruntime = NativeOnnxRuntime(
                root / "onnxruntime",
                root / "onnxruntime/include",
                root / "onnxruntime/lib/libonnxruntime.so",
                "test",
            )
            output = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "WM_PROJECT_VERSION": "2512",
                        "WM_OPTIONS": "linux64GccDPInt32",
                    },
                    clear=False,
                ),
                patch(
                    "foamnordic._cli.shutil.which",
                    side_effect=lambda value: f"/usr/bin/{value}",
                ),
                patch(
                    "foamnordic.execution.runtime_paths.shutil.which",
                    return_value="/usr/bin/wmake",
                ),
                patch(
                    "foamnordic._cli.resolve_onnxruntime",
                    return_value=onnxruntime,
                ),
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "build",
                        "--source",
                        str(repository),
                        "--build-dir",
                        str(build_dir),
                        "--prefix",
                        str(prefix),
                        "--jobs",
                        "2",
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn("[Step 1/6]", output.getvalue())
            self.assertIn("Configure native SDK", output.getvalue())
            self.assertIn("foamnordic_closure_worker", output.getvalue())
            self.assertIn("Install ONNX ClosureHost", output.getvalue())
            self.assertIn("Build OpenFOAM integration", output.getvalue())
            self.assertIn("Build progress-variable solver", output.getvalue())
            self.assertFalse(build_dir.exists())
            self.assertFalse(prefix.exists())

    def test_build_refreshes_only_marker_owned_cache_from_another_source(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            prefix = root / "runtime"
            mark_generated(build_dir, kind="build")
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_HOME_DIRECTORY:INTERNAL=/old/foamnordic/source\n",
                encoding="utf-8",
            )
            output = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "WM_PROJECT_VERSION": "2512",
                        "WM_OPTIONS": "linux64GccDPInt32",
                    },
                    clear=False,
                ),
                patch(
                    "foamnordic._cli.shutil.which",
                    side_effect=lambda value: f"/usr/bin/{value}",
                ),
                patch(
                    "foamnordic.execution.runtime_paths.shutil.which",
                    return_value="/usr/bin/wmake",
                ),
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "build",
                        "--source",
                        str(repository),
                        "--build-dir",
                        str(build_dir),
                        "--prefix",
                        str(prefix),
                        "--without-onnx",
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn("Would refresh build cache", output.getvalue())
            self.assertTrue((build_dir / "CMakeCache.txt").is_file())

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_build_outside_source_requests_a_checkout(self) -> None:
        previous = Path.cwd()
        output = StringIO()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with (
                    patch("foamnordic._cli._source_root", return_value=None),
                    redirect_stdout(output),
                ):
                    status = main(["build"])
        finally:
            os.chdir(previous)
        self.assertEqual(status, 1)

    def test_clobber_removes_only_valid_marker_owned_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            owned = workspace / "runs/owned"
            unowned = workspace / "runs/unowned"
            mark_generated(owned, kind="run")
            unowned.mkdir(parents=True)
            (unowned / MARKER).write_text(
                '{"schema_version": 1, "kind": "run", "root": "/copied"}\n',
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                status = main(["clobber", "--workspace", str(workspace)])
            self.assertEqual(status, 0)
            self.assertFalse(owned.exists())
            self.assertTrue(unowned.exists())
            self.assertIn("Removed 1", output.getvalue())

    def test_clobber_dry_run_preserves_marked_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            owned = workspace / "runs/owned"
            mark_generated(owned, kind="run")
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    ["clobber", "--workspace", str(workspace), "--dry-run"]
                )
            self.assertEqual(status, 0)
            self.assertTrue(owned.exists())
            self.assertIn("would remove", output.getvalue())

    def test_clobber_removes_only_explicit_marker_owned_virtual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            virtual = Path(directory) / "Virtual/FoamNordic"
            mark_generated(virtual, kind="virtual-environment")
            output = StringIO()
            with redirect_stdout(output):
                status = main(["clobber", "--virtual", str(virtual)])
            self.assertEqual(status, 0)
            self.assertFalse(virtual.exists())
            self.assertIn("virtual-environment", output.getvalue())

    def test_clobber_discovers_all_marker_owned_abi_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            runtime = home / ".local/share/foamnordic/runtime/linux-x86_64/openfoam-v2512"
            build = home / ".cache/foamnordic/build/linux-x86_64/openfoam-v2512"
            unowned = home / ".local/share/foamnordic/runtime/linux-x86_64/preserve"
            mark_generated(runtime, kind="native-runtime")
            mark_generated(build, kind="build")
            unowned.mkdir(parents=True)
            with (
                patch("pathlib.Path.home", return_value=home),
                patch.dict(os.environ, {}, clear=True),
                redirect_stdout(StringIO()),
            ):
                status = main(["clobber"])
            self.assertEqual(status, 0)
            self.assertFalse(runtime.exists())
            self.assertFalse(build.exists())
            self.assertTrue(unowned.exists())


if __name__ == "__main__":
    unittest.main()
