from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
from io import StringIO
from foamnordic.build.generations import complete, fingerprint, publish, dependency_identity


class GenerationTests(unittest.TestCase):
    def test_publish_preserves_previous_and_detects_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "generations/one"
            first.mkdir(parents=True)
            (first / "model").write_bytes(b"old")
            publish(root, first, "one")
            self.assertTrue(complete(root / "current", "one"))
            second = root / "generations/two"
            second.mkdir()
            (second / "model").write_bytes(b"new")
            self.assertEqual((root / "current/model").read_bytes(), b"old")
            publish(root, second, "two")
            self.assertEqual((first / "model").read_bytes(), b"old")
            self.assertTrue(complete(root / "current", "two"))
            (second / "model").write_bytes(b"damaged")
            self.assertFalse(complete(root / "current", "two"))

    def test_fingerprint_changes_with_source_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text("one")
            a = fingerprint(root, {"abi": "a"})
            self.assertNotEqual(a, fingerprint(root, {"abi": "b"}))
            (root / "CMakeLists.txt").write_text("two")
            self.assertNotEqual(a, fingerprint(root, {"abi": "a"}))

    def test_failed_publication_preserves_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "one", root / "two"
            first.mkdir(); second.mkdir()
            (first / "file").write_text("old")
            (second / "file").write_text("new")
            publish(root, first, "one")
            with patch("foamnordic.build.generations.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    publish(root, second, "two")
            self.assertEqual((root / "current").resolve(), first.resolve())
            self.assertEqual(list(root.glob(".current-*")), [])

    def test_second_build_skips_compiler_and_failure_keeps_generation(self):
        from foamnordic import _cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("source")
            args = Namespace(dry_run=False, source=source, prefix=root / "runtime",
                             build_dir=root / "build", without_onnx=True)
            selected = Namespace(platform="test", openfoam="v1", build_dir=args.build_dir)
            def build(options, stream):
                options.prefix.mkdir(parents=True)
                (options.prefix / "file").write_text("complete")
                return 0
            with patch.object(_cli, "_source_root", return_value=source), \
                 patch.object(_cli.shutil, "which", return_value="/usr/bin/true"), \
                 patch.object(_cli, "profile", return_value=selected), \
                 patch.object(_cli, "_native_build_environment", return_value={}), \
                 patch.object(_cli, "_build_direct", side_effect=build) as compiler:
                self.assertEqual(_cli._build(args, StringIO()), 0)
                current = (args.prefix / "current").resolve()
                self.assertEqual(_cli._build(args, StringIO()), 0)
                self.assertEqual(compiler.call_count, 1)
                (source / "CMakeLists.txt").write_text("changed")
                compiler.side_effect = RuntimeError("compiler failed")
                with self.assertRaises(RuntimeError):
                    _cli._build(args, StringIO())
                self.assertEqual((args.prefix / "current").resolve(), current)

    def test_dependency_change_invalidates_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            header = root / "src/model.H"
            header.write_text("old")
            environment = {"WM_PROJECT_DIR": str(root)}
            before = dependency_identity(environment, None)
            header.write_text("new version")
            self.assertNotEqual(before, dependency_identity(environment, None))
