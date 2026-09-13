from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import foamnordic as fno


def field_text(*, internal: str = "uniform (1 0 0)") -> str:
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField {internal};
boundaryField
{{
    movingWall
    {{
        type fixedValue;
        value uniform (1 0 0);
    }}
    fixedWalls
    {{
        type noSlip;
    }}
}}
"""


class ForeignPath:
    def __init__(self, value: Path) -> None:
        self.value = value

    def __fspath__(self) -> str:
        return os.fspath(self.value)


def test_initialize_declares_isolated_mesh_preparation() -> None:
    case = fno.OpenFOAM.Case(case_dir="case", run_dir="runs")

    returned = case.initialize(ranks=8, mesh="blockMesh", validate_mesh=True)

    assert returned is case
    assert case.ranks == 8
    assert case.to_plan()["initialization"] == {
        "mesh": "blockMesh",
        "validate_mesh": True,
    }


def test_initialize_rejects_unknown_mesh_generator() -> None:
    case = fno.OpenFOAM.Case(case_dir="case", run_dir="runs")
    with pytest.raises(ValueError, match="blockMesh"):
        case.initialize(mesh="snappyHexMesh")


def test_case_accepts_string_pathlib_and_foreign_pathlike() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "case/0").mkdir(parents=True)
        (root / "case/0/U").write_text(field_text(), encoding="utf-8")
        case = fno.OpenFOAM.Case(
            case_dir=ForeignPath(root / "case"),
            run_dir=str(root / "workspace"),
        )
        assert isinstance(case.case_dir, Path)
        assert isinstance(case.run_dir, Path)
        assert tuple(case.fields) == ("U",)
        assert case.field("U").field_class == "volVectorField"
        assert case.field("U").boundary_names == ("movingWall", "fixedWalls")
        assert case.boundary_names == ("fixedWalls", "movingWall")


def test_native_directive_uses_declared_toolchain_expansion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case_root = root / "case"
        tools = root / "tools"
        (case_root / "0").mkdir(parents=True)
        tools.mkdir()
        (case_root / "0/U").write_text(
            field_text(internal='#calc "1 + 1"'), encoding="utf-8"
        )
        foam_dictionary = tools / "foamDictionary"
        expanded = field_text(internal="uniform (2 0 0)")
        foam_dictionary.write_text(
            "#!/bin/sh\ncat <<'EOF'\n" + expanded + "EOF\n",
            encoding="utf-8",
        )
        foam_dictionary.chmod(0o750)
        case = fno.OpenFOAM.Case(
            case_dir=case_root,
            run_dir=root / "workspace",
            of_cmd=f"export PATH={tools}:$PATH",
            shell="bash",
        )
        value = case.field("U").internal_value
        assert "2" in str(value)


def test_grouped_namespace_naming_is_canonical_and_compatible() -> None:
    assert fno.OpenFOAM is fno.openfoam
    assert fno.Export is fno.export
    assert fno.Models is fno.models
    assert fno.Runtime is fno.runtime
    assert "Case" in dir(fno.OpenFOAM)
    assert "Toolchain" not in dir(fno.OpenFOAM)
    assert "openfoam" not in dir(fno)
    assert "field" not in dir(fno)
    assert "Result" not in dir(fno)
    assert "Result" in dir(fno.Runtime)
    assert "CompiledPlan" in dir(fno.Runtime)


def test_module_shorthand_is_lowered_to_internal_environment() -> None:
    case = fno.OpenFOAM.Case(
        case_dir="case",
        run_dir="runs",
        of_cmd="openfoam/2512",
        shell="zsh",
    )
    assert case.of_cmd == "openfoam/2512"
    assert case.shell == "zsh"
    assert case._toolchain.command == "module load openfoam/2512"


def test_macos_openfoam_command_is_used_as_a_wrapper() -> None:
    case = fno.OpenFOAM.Case(
        case_dir="case",
        run_dir="runs",
        of_cmd="openfoam",
        shell="zsh",
    )
    assert case._toolchain.command == "openfoam"
    assert case._toolchain.wrapper

    from foamnordic.execution.shell import toolchain_shell

    command = toolchain_shell(case._toolchain, "exec pimpleFoam -help")
    assert command[:2] == ("zsh", "-lc")
    assert "openfoam -c" in command[2]
    assert "exec pimpleFoam -help" in command[2]


def test_macos_wrapper_contains_local_mpi_solver() -> None:
    from foamnordic.execution.launch import _solver_command

    case = fno.OpenFOAM.Case(
        name="lidDrivenCavity",
        case_dir="case",
        run_dir="runs",
        of_cmd="openfoam",
        shell="zsh",
        application="pimpleFoam",
        ranks=6,
    )
    command = _solver_command(
        fno.Longship(case=case),
        Path("prepared-case"),
        local_mpi=True,
    )
    assert command[:2] == ("zsh", "-lc")
    assert "openfoam -c" in command[2]
    assert "-np 6 pimpleFoam" in command[2]


def test_macos_explicit_mpi_launcher_isolated_from_openfoam_environment(mocker) -> None:
    from foamnordic.execution.launch import _solver_command

    with tempfile.TemporaryDirectory() as temporary:
        launcher = Path(temporary) / "mpirun"
        launcher.touch(mode=0o755)
        case = fno.OpenFOAM.Case(
            name="cavity",
            case_dir="case",
            run_dir="runs",
            of_cmd="openfoam -prefix=/Volumes/OpenFOAM-v2606",
            shell="zsh",
            application="pimpleFoam",
            ranks=6,
        )
        _mock_patch_1 = mocker.patch.dict(
            os.environ, {"FOAMNORDIC_MPIRUN": str(launcher)}
        )
        _mock_patch_2 = mocker.patch(
            "foamnordic.execution.mpi.platform.system", return_value="Darwin"
        )
        try:
            command = _solver_command(
                fno.Longship(case=case),
                Path("prepared-case"),
                local_mpi=True,
            )
        finally:
            mocker.stop(_mock_patch_2)
            mocker.stop(_mock_patch_1)

    assert str(launcher) in command[2]
    assert "unset DYLD_LIBRARY_PATH OPAL_PREFIX MPI_ARCH_PATH" in command[2]
    assert "-x DYLD_LIBRARY_PATH=" in command[2]
    assert "-np 6 pimpleFoam" in command[2]
    assert "-parallel" in command[2]


@pytest.mark.parametrize("of_cmd", ["module load openfoam/2512", "openfoam"])
def test_linux_explicit_mpi_launcher_preserves_openfoam_environment(
    of_cmd, mocker
) -> None:
    from foamnordic.execution.launch import _solver_command

    with tempfile.TemporaryDirectory() as temporary:
        launcher = Path(temporary) / "mpirun"
        launcher.touch(mode=0o755)
        case = fno.OpenFOAM.Case(
            name="cavity",
            case_dir="case",
            run_dir="runs",
            of_cmd=of_cmd,
            shell="bash",
            application="pimpleFoam",
            ranks=6,
        )
        _mock_patch_3 = mocker.patch.dict(
            os.environ, {"FOAMNORDIC_MPIRUN": str(launcher)}
        )
        _mock_patch_4 = mocker.patch(
            "foamnordic.execution.mpi.platform.system", return_value="Linux"
        )
        try:
            command = _solver_command(
                fno.Longship(case=case),
                Path("prepared-case"),
                local_mpi=True,
            )
        finally:
            mocker.stop(_mock_patch_4)
            mocker.stop(_mock_patch_3)

        assert str(launcher) in command[2]
        assert "unset DYLD_LIBRARY_PATH" not in command[2]
        assert "-x DYLD_LIBRARY_PATH=" not in command[2]
        assert "-np 6 pimpleFoam" in command[2]
        assert "-parallel" in command[2]
