import re
from pathlib import Path


def test_bindnordic_does_not_depend_on_foamnordic() -> None:
    repository = Path(__file__).resolve().parents[3]
    bindnordic = repository / "packages/bindnordic"
    patterns = (
        re.compile(r"^\s*(?:from|import)\s+foamnordic\b", re.MULTILINE),
        re.compile(r"#include\s*[<\"]foamnordic/"),
        re.compile(r"FoamNordic::"),
    )
    offenders = []
    for path in sorted(bindnordic.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".C",
            ".H",
            ".cmake",
            ".cpp",
            ".hpp",
            ".py",
            ".txt",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            offenders.append(str(path.relative_to(repository)))
    assert offenders == [], "BindNordic dependency boundary was violated"
