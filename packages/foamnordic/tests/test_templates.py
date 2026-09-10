from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foamnordic.execution import templates


class TemplateLookupTests(unittest.TestCase):
    def test_values_are_rendered_without_interpreting_braces(self) -> None:
        self.assertEqual(
            templates.render(
                "path @PATH@; count @COUNT@;",
                {"PATH": "observations.{rank}.jsonl", "COUNT": 4},
                kind="test",
            ),
            "path observations.{rank}.jsonl; count 4;",
        )

    def test_unresolved_tokens_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "@MISSING@"):
            templates.render("value @MISSING@;", {}, kind="test")

    def test_packaged_template_precedes_source_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packaged = root / "package/templates/openfoam/example.in"
            source = root / "tools/templates/openfoam/example.in"
            packaged.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            packaged.write_text("packaged", encoding="utf-8")
            source.write_text("source", encoding="utf-8")

            with (
                patch.object(templates, "_REPOSITORY_ROOT", root),
                patch.object(templates, "files", return_value=root / "package"),
            ):
                value = templates.read("openfoam/example.in", kind="test")

        self.assertEqual(value, "packaged")

    def test_source_checkout_is_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tools/templates/slurm/example.in"
            source.parent.mkdir(parents=True)
            source.write_text("source", encoding="utf-8")

            with (
                patch.object(templates, "_REPOSITORY_ROOT", root),
                patch.object(templates, "files", return_value=root / "package"),
            ):
                value = templates.read("slurm/example.in", kind="test")

        self.assertEqual(value, "source")

    def test_missing_optional_template_uses_its_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(templates, "_REPOSITORY_ROOT", root),
                patch.object(templates, "files", return_value=root / "package"),
            ):
                value = templates.read(
                    "large_banner.txt",
                    kind="test",
                    default="FoamNordic",
                )

        self.assertEqual(value, "FoamNordic")

    def test_template_path_cannot_escape_its_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the template directory"):
            templates.read("../secret", kind="test")


if __name__ == "__main__":
    unittest.main()
