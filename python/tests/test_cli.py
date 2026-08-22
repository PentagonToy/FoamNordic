from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest

import foamnordic as fno
from foamnordic._cli import main
from foamnordic._managed import MARKER, mark_generated
from foamnordic._native_plan import available as native_available


class CliTests(unittest.TestCase):
    def test_public_version_and_directory_are_discoverable(self) -> None:
        self.assertEqual(fno.__version__, "1.0.3.dev4")
        self.assertIn("Longship", dir(fno))
        self.assertIn("export", dir(fno))

    def test_cli_version_is_discoverable(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as stopped, redirect_stdout(output):
            main(["--version"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "foamnordic 1.0.3.dev4")

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

    def test_source_build_dry_run_is_visual_and_non_mutating(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            prefix = root / "install"
            output = StringIO()
            with redirect_stdout(output):
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
            self.assertIn("[Step 1/3]", output.getvalue())
            self.assertIn("Configure native SDK", output.getvalue())
            self.assertFalse(build_dir.exists())
            self.assertFalse(prefix.exists())

    @unittest.skipUnless(native_available(), "nanobind extension is not installed")
    def test_binary_wheel_build_is_a_successful_noop(self) -> None:
        previous = Path.cwd()
        output = StringIO()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with redirect_stdout(output):
                    status = main(["build"])
        finally:
            os.chdir(previous)
        self.assertEqual(status, 0)
        self.assertIn("requires no native core rebuild", output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
