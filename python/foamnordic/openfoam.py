"""Immutable OpenFOAM case declarations."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
import shlex
from types import MappingProxyType
from typing import Mapping

from ._openfoam_reader import Field, read_case_fields
from ._paths import PathInput, path_from
from ._validation import require_nonempty, require_positive


@dataclass(frozen=True, slots=True)
class _Toolchain:
    """Shell command that prepares OpenFOAM before a native command runs.

    The command is evaluated by the selected login shell for each generated
    build or launch script. It does not attempt to mutate the notebook's
    current process environment.
    """

    command: str | None = None
    shell: str = "bash"
    wrapper: bool = False

    def __post_init__(self) -> None:
        if self.command is not None:
            object.__setattr__(
                self,
                "command",
                require_nonempty(self.command, "toolchain command"),
            )
        shell = require_nonempty(self.shell, "toolchain shell")
        if Path(shell).name not in {"bash", "zsh"}:
            raise ValueError("shell must resolve to bash or zsh")
        object.__setattr__(self, "shell", shell)

    @classmethod
    def module(cls, name: str, *, shell: str = "bash") -> "_Toolchain":
        """Create a toolchain that loads one environment module."""

        module_name = require_nonempty(name, "module name")
        return cls(command=f"module load {shlex.quote(module_name)}", shell=shell)

    @classmethod
    def openfoam_wrapper(
        cls, command: str = "openfoam", *, shell: str = "zsh"
    ) -> "_Toolchain":
        """Run commands through the OpenFOAM.app ``openfoam`` wrapper."""

        return cls(
            command=require_nonempty(command, "OpenFOAM wrapper command"),
            shell=shell,
            wrapper=True,
        )

    def to_plan(self) -> dict[str, str | bool | None]:
        return {
            "command": self.command,
            "shell": self.shell,
            "wrapper": self.wrapper,
        }


@dataclass(frozen=True, slots=True)
class DictionaryTemplate:
    """Custom OpenFOAM dictionary template for a closure integration.

    FoamNordic supplies transport and field-contract variables. Application-
    specific variables allow combustion and other solver plugins to render
    their own dictionary without changing the Longship launcher.
    """

    source: PathInput
    destination: PathInput
    variables: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        source = path_from(self.source)
        destination = path_from(self.destination, expand_user=False)
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError("dictionary destination must stay inside the case")
        if not destination.parts:
            raise ValueError("dictionary destination must not be empty")
        normalized = {
            require_nonempty(str(name), "template variable"): str(value)
            for name, value in dict(self.variables).items()
        }
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "variables", MappingProxyType(normalized))

    def to_plan(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "variables": dict(self.variables),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Case:
    """OpenFOAM case, output location, and shell environment declaration.

    A compact ``of_cmd`` value such as ``openfoam/2512`` names an environment
    module. ``openfoam`` uses the OpenFOAM.app command wrapper on macOS. A full
    shell statement such as ``source /opt/OpenFOAM/bashrc`` is executed
    verbatim. ``None`` uses OpenFOAM commands already available on ``PATH``.
    """

    name: str | None = None
    case_dir: PathInput
    run_dir: PathInput
    of_cmd: str | None = None
    shell: str = "bash"
    application: str = "pimpleFoam"
    ranks: int = 1
    integration: DictionaryTemplate | None = None
    _fields: dict[str, Field] = dataclass_field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _boundary_names: tuple[str, ...] = dataclass_field(
        default=(), init=False, repr=False, compare=False
    )
    _fields_loaded: bool = dataclass_field(
        default=False, init=False, repr=False, compare=False
    )
    _toolchain: _Toolchain = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        case_dir = path_from(self.case_dir)
        run_dir = path_from(self.run_dir)
        if case_dir == run_dir:
            raise ValueError("case_dir and run_dir must be different paths")
        object.__setattr__(self, "case_dir", case_dir)
        object.__setattr__(self, "run_dir", run_dir)
        object.__setattr__(
            self,
            "name",
            require_nonempty(self.name or case_dir.name, "case name"),
        )
        object.__setattr__(
            self,
            "application",
            require_nonempty(self.application, "application"),
        )
        require_positive(self.ranks, "ranks")
        command = self.of_cmd
        if command is not None:
            command = require_nonempty(command, "of_cmd")
        words = () if command is None else tuple(shlex.split(command))
        if words and Path(words[0]).name == "openfoam":
            toolchain = _Toolchain.openfoam_wrapper(command, shell=self.shell)
        elif command is not None and "/" in command and not any(
            character.isspace() for character in command
        ):
            toolchain = _Toolchain.module(command, shell=self.shell)
        else:
            toolchain = _Toolchain(command=command, shell=self.shell)
        object.__setattr__(self, "of_cmd", command)
        object.__setattr__(self, "shell", toolchain.shell)
        object.__setattr__(self, "_toolchain", toolchain)

    def _load_fields(self) -> None:
        if self._fields_loaded:
            return
        fields, boundaries = read_case_fields(self.initial_directory, self._toolchain)
        self._fields.update(fields)
        object.__setattr__(self, "_boundary_names", boundaries)
        object.__setattr__(self, "_fields_loaded", True)

    @property
    def initial_directory(self) -> Path:
        """Return ``0/`` when present, otherwise the conventional ``0.orig/``."""

        initial = self.case_dir / "0"
        if initial.is_dir():
            return initial
        original = self.case_dir / "0.orig"
        if original.is_dir():
            return original
        raise FileNotFoundError(
            f"OpenFOAM case has neither 0 nor 0.orig: {self.case_dir}"
        )

    @property
    def fields(self) -> Mapping[str, Field]:
        """Initial fields discovered from ``0/`` without mutating the case."""

        self._load_fields()
        return MappingProxyType(self._fields)

    @property
    def boundary_names(self) -> tuple[str, ...]:
        """Unique initial-field boundary names in deterministic order."""

        self._load_fields()
        return self._boundary_names

    def field(self, name: str) -> Field:
        """Return one initial field with an informative missing-field error."""

        field_name = require_nonempty(name, "field name")
        try:
            return self.fields[field_name]
        except KeyError:
            available = ", ".join(self.fields) or "none"
            raise KeyError(
                f"OpenFOAM field {field_name!r} was not found; available: {available}"
            ) from None

    def to_plan(self) -> dict[str, object]:
        return {
            "name": self.name,
            "case_dir": str(self.case_dir),
            "run_dir": str(self.run_dir),
            "of_cmd": self.of_cmd,
            "shell": self.shell,
            "application": self.application,
            "ranks": self.ranks,
            "integration": (
                None if self.integration is None else self.integration.to_plan()
            ),
        }


__all__ = ["Case", "DictionaryTemplate", "Field"]


def __dir__() -> list[str]:
    return sorted(__all__)
