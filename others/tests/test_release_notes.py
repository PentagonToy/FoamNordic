import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseNotesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for filename, text in {
            "python/pyproject.toml": 'version = "1.0.5"\n',
            "python/foamnordic/__init__.py": '__version__ = "1.0.5"\n',
            "CMakeLists.txt": 'project(FoamNordic VERSION 1.0.5 LANGUAGES CXX)\n',
            "python/buildkit/CMakeLists.txt": 'project(FoamNordicBuildKit VERSION 1.0.5 LANGUAGES CXX)\n',
            "CHANGELOG.md": '# Changelog\n\n## [Unreleased]\nFuture\n\n## [1.0.5]\n\nExact body.\n\n### Notes\nKeep formatting.\n\n## [1.0.4]\nOld body.\n',
        }.items():
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def test_exact_section(self):
        self.assertEqual(module.release_notes(self.root, "v1.0.5"),
                         "Exact body.\n\n### Notes\nKeep formatting.\n")

    def test_tag_mismatch(self):
        with self.assertRaises(ValueError):
            module.release_notes(self.root, "v1.0.4")

    def test_version_mismatch(self):
        (self.root / "python/pyproject.toml").write_text('version = "1.0.4"\n')
        with self.assertRaises(ValueError):
            module.release_notes(self.root)

    def test_missing_empty_duplicate(self):
        for text in ("# Changelog\n", "## [1.0.5]\n", "## [1.0.5]\nA\n## [1.0.5]\nB\n"):
            with self.subTest(text=text):
                (self.root / "CHANGELOG.md").write_text(text)
                with self.assertRaises(ValueError):
                    module.release_notes(self.root)

    def test_repository(self):
        self.assertIn("1.0.5", module.release_notes(SCRIPT.parent.parent))


if __name__ == "__main__":
    unittest.main()
