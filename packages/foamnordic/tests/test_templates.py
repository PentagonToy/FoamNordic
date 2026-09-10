from __future__ import annotations

import unittest

from foamnordic.execution.templates import render


class TemplateRenderingTests(unittest.TestCase):
    def test_values_are_rendered_without_interpreting_braces(self) -> None:
        self.assertEqual(
            render(
                "path @PATH@; count @COUNT@;",
                {"PATH": "observations.{rank}.jsonl", "COUNT": 4},
                kind="test",
            ),
            "path observations.{rank}.jsonl; count 4;",
        )

    def test_unresolved_tokens_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "@MISSING@"):
            render("value @MISSING@;", {}, kind="test")


if __name__ == "__main__":
    unittest.main()
