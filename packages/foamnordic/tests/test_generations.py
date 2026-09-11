import tempfile
from argparse import Namespace
from io import StringIO
from pathlib import Path

import pytest
from foamnordic.build.generations import (
    collect_garbage,
    complete,
    dependency_identity,
    fingerprint,
    pin,
    publish,
    source_identity,
)


def test_gc_preserves_current_recent_and_pinned_generations():
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
        assert set(removed) == {generations[1]}
        assert generations[0].is_dir()
        assert generations[2].is_dir()
        assert generations[3].is_dir()
        marker.unlink()
        assert collect_garbage(root, keep=2) == (generations[0],)


def test_publish_preserves_previous_and_detects_damage():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "generations/one"
        first.mkdir(parents=True)
        (first / "model").write_bytes(b"old")
        publish(root, first, "one")
        assert complete(root / "current", "one")
        second = root / "generations/two"
        second.mkdir()
        (second / "model").write_bytes(b"new")
        assert (root / "current/model").read_bytes() == b"old"
        publish(root, second, "two")
        assert (first / "model").read_bytes() == b"old"
        assert complete(root / "current", "two")
        (second / "model").write_bytes(b"damaged")
        assert not complete(root / "current", "two")


def test_fingerprint_changes_with_source_and_options():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "CMakeLists.txt").write_text("one")
        a = fingerprint(root, {"abi": "a"})
        assert a != fingerprint(root, {"abi": "b"})
        (root / "CMakeLists.txt").write_text("two")
        assert a != fingerprint(root, {"abi": "a"})


def test_failed_publication_preserves_current(mocker):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first, second = root / "one", root / "two"
        first.mkdir()
        second.mkdir()
        (first / "file").write_text("old")
        (second / "file").write_text("new")
        publish(root, first, "one")
        _mock_patch_1 = mocker.patch(
            "foamnordic.build.generations.os.replace",
            side_effect=OSError("interrupted"),
        )
        try:
            with (
                pytest.raises(OSError),
            ):
                publish(root, second, "two")
        finally:
            mocker.stop(_mock_patch_1)
        assert (root / "current").resolve() == first.resolve()
        assert list(root.glob(".current-*")) == []


def test_second_build_skips_compiler_and_failure_keeps_generation(mocker):
    from foamnordic import _cli

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text("source")
        args = Namespace(
            dry_run=False,
            source=source,
            prefix=root / "runtime",
            build_dir=root / "build",
            without_onnx=True,
        )
        selected = Namespace(platform="test", openfoam="v1", build_dir=args.build_dir)

        def build(options, stream):
            options.prefix.mkdir(parents=True)
            (options.prefix / "file").write_text("complete")
            return 0

        _mock_patch_2 = mocker.patch.object(_cli, "_source_root", return_value=source)
        _mock_patch_3 = mocker.patch.object(
            _cli.shutil, "which", return_value="/usr/bin/true"
        )
        _mock_patch_4 = mocker.patch.object(_cli, "profile", return_value=selected)
        _mock_patch_5 = mocker.patch.object(
            _cli, "_native_build_environment", return_value={}
        )
        compiler = mocker.patch.object(_cli, "_build_direct", side_effect=build)
        try:
            assert _cli._build(args, StringIO()) == 0
            current = (args.prefix / "current").resolve()
            assert _cli._build(args, StringIO()) == 0
            assert compiler.call_count == 1
            (source / "CMakeLists.txt").write_text("changed")
            compiler.side_effect = RuntimeError("compiler failed")
            with pytest.raises(RuntimeError):
                _cli._build(args, StringIO())
            assert (args.prefix / "current").resolve() == current
        finally:
            mocker.stop(compiler)
            mocker.stop(_mock_patch_5)
            mocker.stop(_mock_patch_4)
            mocker.stop(_mock_patch_3)
            mocker.stop(_mock_patch_2)


def test_dependency_change_invalidates_identity():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        header = root / "src/model.H"
        header.write_text("old")
        environment = {"WM_PROJECT_DIR": str(root)}
        before = dependency_identity(environment, None)
        header.write_text("new version")
        assert before != dependency_identity(environment, None)


def test_external_source_change_invalidates_identity():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model.cpp").write_text("old")
        before = source_identity(root)
        (root / "model.cpp").write_text("new")
        assert before != source_identity(root)
