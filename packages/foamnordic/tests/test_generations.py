from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
from io import StringIO
from foamnordic.build.generations import (
    collect_garbage,
    complete,
    dependency_identity,
    fingerprint,
    pin,
    publish,
    source_identity,
)


class GenerationTests(unittest.TestCase):
    def test_gc_preserves_current_recent_and_pinned_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = []
            for index in range(4):
                generation = root / "generations" / str(index)
                generation.mkdir(parents=True)
                (generation / "runtime").write_text(str(index))
                publish(root, generation, str(index))
                generations.append(generation)
            marker = pin(generations[0], "active run")
            removed = collect_garbage(root, keep=2)
            self.assertEqual(set(removed), {generations[1]})
            self.assertTrue(generations[0].is_dir())
            self.assertTrue(generations[2].is_dir())
            self.assertTrue(generations[3].is_dir())
            marker.unlink()
            self.assertEqual(collect_garbage(root, keep=2), (generations[0],))

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

    def test_external_source_change_invalidates_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.cpp").write_text("old")
            before = source_identity(root)
            (root / "model.cpp").write_text("new")
            self.assertNotEqual(before, source_identity(root))
