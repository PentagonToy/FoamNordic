from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from foamnordic.execution import templates


def test_values_are_rendered_without_interpreting_braces() -> None:
    assert (
        templates.render(
            "path @PATH@; count @COUNT@;",
            {"PATH": "observations.{rank}.jsonl", "COUNT": 4},
            kind="test",
        )
        == "path observations.{rank}.jsonl; count 4;"
    )


def test_unresolved_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="@MISSING@"):
        templates.render("value @MISSING@;", {}, kind="test")


def test_packaged_template_precedes_source_fallback(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        packaged = root / "package/templates/openfoam/example.in"
        source = root / "tools/templates/openfoam/example.in"
        packaged.parent.mkdir(parents=True)
        source.parent.mkdir(parents=True)
        packaged.write_text("packaged", encoding="utf-8")
        source.write_text("source", encoding="utf-8")
        _mock_patch_1 = mocker.patch.object(templates, "_REPOSITORY_ROOT", root)
        _mock_patch_2 = mocker.patch.object(
            templates, "files", return_value=root / "package"
        )
        try:
            value = templates.read("openfoam/example.in", kind="test")
        finally:
            mocker.stop(_mock_patch_2)
            mocker.stop(_mock_patch_1)

    assert value == "packaged"


def test_source_checkout_is_the_fallback(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "tools/templates/slurm/example.in"
        source.parent.mkdir(parents=True)
        source.write_text("source", encoding="utf-8")
        _mock_patch_3 = mocker.patch.object(templates, "_REPOSITORY_ROOT", root)
        _mock_patch_4 = mocker.patch.object(
            templates, "files", return_value=root / "package"
        )
        try:
            value = templates.read("slurm/example.in", kind="test")
        finally:
            mocker.stop(_mock_patch_4)
            mocker.stop(_mock_patch_3)

    assert value == "source"


def test_missing_optional_template_uses_its_default(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _mock_patch_5 = mocker.patch.object(templates, "_REPOSITORY_ROOT", root)
        _mock_patch_6 = mocker.patch.object(
            templates, "files", return_value=root / "package"
        )
        try:
            value = templates.read(
                "large_banner.txt",
                kind="test",
                default="FoamNordic",
            )
        finally:
            mocker.stop(_mock_patch_6)
            mocker.stop(_mock_patch_5)

    assert value == "FoamNordic"


def test_template_path_cannot_escape_its_root() -> None:
    with pytest.raises(ValueError, match="inside the template directory"):
        templates.read("../secret", kind="test")
