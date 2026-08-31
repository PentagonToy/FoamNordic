"""Lazy OpenFOAM field metadata reading with native directive expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

from .core.paths import PathInput, path_from
from .execution.shell import quote_command, toolchain_shell

if TYPE_CHECKING:
    from .openfoam import _Toolchain


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


class _DictionaryReader:
    def __init__(self, path: PathInput, toolchain: _Toolchain) -> None:
        self.path = path_from(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"OpenFOAM dictionary was not found: {self.path}")
        try:
            from foamlib import FoamFile
        except ImportError as error:
            raise ImportError("OpenFOAM field discovery requires foamlib") from error
        self._foam_file = FoamFile(self.path)
        self._toolchain = toolchain
        self._resolved = None
        self._temporary: Path | None = None
        self._needs_native = bool(_NATIVE_DIRECTIVE.search(self.path.read_bytes()))

    def get(self, entry: str, default=None):
        if self._resolved is not None:
            return self._resolved.get(entry, default)
        if self._needs_native:
            return self._resolve().get(entry, default)
        try:
            return self._foam_file.get(entry, default)
        except Exception as error:
            if type(error).__name__ != "FoamFileDecodeError":
                raise
            return self._resolve().get(entry, default)

    def _resolve(self):
        if self._resolved is not None:
            return self._resolved
        command = quote_command(("foamDictionary", self.path, "-expand"))
        result = subprocess.run(
            toolchain_shell(self._toolchain, command),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"foamDictionary failed to expand {self.path}: {message}"
            )
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="foamnordic-", suffix=f"-{self.path.name}",
            delete=False,
        )
        try:
            temporary.write(result.stdout)
        finally:
            temporary.close()
        self._temporary = Path(temporary.name)
        from foamlib import FoamFile

        self._resolved = FoamFile(self._temporary)
        return self._resolved

    def close(self) -> None:
        self._resolved = None
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def read_field(path: PathInput, toolchain: _Toolchain) -> Field | None:
    """Read one OpenFOAM field, expanding native directives when necessary."""

    field_path = path_from(path).resolve()
    with _DictionaryReader(field_path, toolchain) as reader:
        header = reader.get("FoamFile")
        boundary = reader.get("boundaryField")
        if header is None or boundary is None:
            return None
        name = str(header.get("object") or field_path.name).strip()
        field_class = str(header.get("class") or "").strip()
        if not name or not field_class:
            return None
        return Field(
            name=name,
            field_class=field_class,
            dimensions=reader.get("dimensions"),
            internal_value=reader.get("internalField"),
            boundary_names=tuple(str(value) for value in boundary),
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
