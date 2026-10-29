import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    for filename, text in {
        "packages/foamnordic/pyproject.toml": 'version = "1.0.5"\n',
        "packages/foamnordic/foamnordic/__init__.py": '__version__ = "1.0.5"\n',
        "packages/bindnordic/pyproject.toml": 'version = "1.0.5"\n',
        "CMakeLists.txt": "project(FoamNordic VERSION 1.0.5 LANGUAGES CXX)\n",
        "packages/foamnordic/buildkit/CMakeLists.txt": "project(FoamNordicBuildKit VERSION 1.0.5 LANGUAGES CXX)\n",
        "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\nFuture\n\n## [1.0.5]\n\nExact body.\n\n### Notes\nKeep formatting.\n\n## [1.0.4]\nOld body.\n",
    }.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_exact_section(release_root: Path) -> None:
    assert module.release_notes(release_root, "v1.0.5") == (
        "Exact body.\n\n### Notes\nKeep formatting.\n"
    )


def test_tag_mismatch(release_root: Path) -> None:
    with pytest.raises(ValueError):
        module.release_notes(release_root, "v1.0.4")


def test_version_mismatch(release_root: Path) -> None:
    (release_root / "packages/foamnordic/pyproject.toml").write_text(
        'version = "1.0.4"\n'
    )
    with pytest.raises(ValueError):
        module.release_notes(release_root)


def test_bindnordic_version_mismatch(release_root: Path) -> None:
    (release_root / "packages/bindnordic/pyproject.toml").write_text(
        'version = "1.0.4"\n'
    )
    with pytest.raises(ValueError):
        module.release_notes(release_root)


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n",
        "## [1.0.5]\n",
        "## [1.0.5]\nA\n## [1.0.5]\nB\n",
    ],
)
def test_missing_empty_or_duplicate_section(release_root: Path, changelog: str) -> None:
    (release_root / "CHANGELOG.md").write_text(changelog)
    with pytest.raises(ValueError):
        module.release_notes(release_root)


def test_repository() -> None:
    notes = module.release_notes(SCRIPT.parents[2])
    assert "Slurm memory calculations" in notes
    assert "`hedgehogs>=0.0.5`" in notes
    assert "## [1.0.9]" not in notes
    assert "`fno.Export.table()`" not in notes
