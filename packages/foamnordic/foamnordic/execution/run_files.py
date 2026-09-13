"""Own generated run paths and initial log contents."""

from __future__ import annotations

from pathlib import Path

from .templates import read as read_template


def safe_name(name: str) -> str:
    value = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-")
    return value[:128] or "FoamNordic"


def sailing_paths(work_dir: Path, name: str) -> tuple[Path, Path, Path]:
    logs = Path(work_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = safe_name(name)
    return (
        logs / f"Sailing_{stem}.log",
        logs / f"Harbor_{stem}.log",
        logs / f"Sailing_{stem}.out",
    )


def internal_path(work_dir: Path, name: str) -> Path:
    internal = Path(work_dir) / ".foamnordic"
    internal.mkdir(parents=True, exist_ok=True)
    return internal / name


def _banner() -> str:
    return read_template(
        "large_banner.txt",
        kind="runtime banner",
        default="FoamNordic",
    ).rstrip()


def initialize_sailing_log(path: Path, name: str) -> None:
    if path.is_file():
        return
    path.write_text(
        f"{_banner()}\n\n[FoamNordic] Sailing: {name}\n",
        encoding="utf-8",
    )


def initialize_harbor_log(path: Path, name: str) -> None:
    path.write_text(
        f"{_banner()}\n\n[FoamNordic] Harbor: {name}\n",
        encoding="utf-8",
    )


__all__ = [
    "initialize_harbor_log",
    "initialize_sailing_log",
    "internal_path",
    "safe_name",
    "sailing_paths",
]
