"""Check release versions and print the matching CHANGELOG section (no publishing)."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def release_notes(root: Path, tag: str | None = None) -> str:
    sources = (
        ("python/pyproject.toml", r'^version = "([^"]+)"$'),
        ("python/foamnordic/__init__.py", r'^__version__ = "([^"]+)"$'),
        ("CMakeLists.txt", r'^project\(FoamNordic VERSION (\S+) LANGUAGES CXX\)$'),
        ("python/buildkit/CMakeLists.txt", r'^project\(FoamNordicBuildKit VERSION (\S+) LANGUAGES CXX\)$'),
    )
    versions = []
    for filename, pattern in sources:
        matches = re.findall(pattern, (root / filename).read_text(), re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Expected one version in {filename}")
        versions.append(matches[0])
    if len(set(versions)) != 1:
        raise ValueError(f"Release versions disagree: {versions}")
    version = versions[0]
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match v{version}")
    text = (root / "CHANGELOG.md").read_text()
    headers = list(re.finditer(r'^## \[([^\]]+)\].*$', text, re.MULTILINE))
    matches = [i for i, header in enumerate(headers) if header[1] == version]
    if len(matches) != 1:
        raise ValueError(f"Expected one CHANGELOG section for {version}")
    index = matches[0]
    end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
    body = text[headers[index].end():end].strip()
    if not body:
        raise ValueError(f"Empty release notes for {version}")
    return body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require this Git tag to match the package version")
    args = parser.parse_args()
    try:
        print(release_notes(ROOT, args.tag), end="")
    except ValueError as error:
        parser.exit(1, f"Release preparation failed: {error}\n")


if __name__ == "__main__":
    main()
