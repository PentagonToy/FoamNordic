"""Lazy OpenFOAM field metadata reading with native directive expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

from bindnordic.openfoam import OpenFOAMDecodeError
from bindnordic.openfoam import read_field as read_openfoam_field

from .paths import PathInput, path_from
from ..execution.shell import quote_command, toolchain_shell

if TYPE_CHECKING:
    from ..openfoam import _Toolchain


_NATIVE_DIRECTIVE = re.compile(
    rb"#(?:include|includeIfPresent|calc|eval|codeStream)\b|\$\{?[A-Za-z_]"
)


@dataclass(frozen=True, slots=True)
class Field:
    """Metadata read from one initial OpenFOAM field."""

    name: str
    field_class: str
    dimensions: object
    internal_value: object
    boundary_names: tuple[str, ...]
    path: Path


def _expanded(path: Path, toolchain: _Toolchain) -> Path | None:
    if not _NATIVE_DIRECTIVE.search(path.read_bytes()):
        return None
    command = quote_command(("foamDictionary", path, "-expand"))
    result = subprocess.run(
        toolchain_shell(toolchain, command),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"foamDictionary failed to expand {path}: {message}")
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="foamnordic-",
        suffix=f"-{path.name}",
        delete=False,
    )
    try:
        temporary.write(result.stdout)
    finally:
        temporary.close()
    return Path(temporary.name)


def read_field(path: PathInput, toolchain: _Toolchain) -> Field | None:
    """Read one OpenFOAM field, expanding native directives when necessary."""

    field_path = path_from(path).resolve()
    if not field_path.is_file():
        raise FileNotFoundError(f"OpenFOAM dictionary was not found: {field_path}")
    temporary = _expanded(field_path, toolchain)
    try:
        value = read_openfoam_field(temporary or field_path)
    except OpenFOAMDecodeError:
        return None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return Field(
        name=value.name,
        field_class=value.field_class,
        dimensions=value.dimensions,
        internal_value=value.values,
        boundary_names=value.boundaries,
        path=field_path,
    )


def read_case_fields(
    time_directory: PathInput, toolchain: _Toolchain
) -> tuple[dict[str, Field], tuple[str, ...]]:
    """Discover valid fields in an OpenFOAM time directory."""

    directory = path_from(time_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"OpenFOAM time directory was not found: {directory}")
    fields: dict[str, Field] = {}
    boundaries: list[str] = []
    for field_path in sorted(directory.iterdir()):
        if not field_path.is_file():
            continue
        payload = field_path.read_bytes()[:65536]
        if b"FoamFile" not in payload:
            continue
        field = read_field(field_path, toolchain)
        if field is None:
            continue
        fields[field.name] = field
        for boundary in field.boundary_names:
            if boundary not in boundaries:
                boundaries.append(boundary)
    return fields, tuple(sorted(boundaries))
