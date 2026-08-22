from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

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


class OpenFOAMReaderTests(unittest.TestCase):
    def test_case_accepts_string_pathlib_and_foreign_pathlike(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case/0").mkdir(parents=True)
            (root / "case/0/U").write_text(field_text(), encoding="utf-8")
            case = fno.OpenFOAM.Case(
                case_dir=ForeignPath(root / "case"),
                run_dir=str(root / "workspace"),
            )
            self.assertIsInstance(case.case_dir, Path)
            self.assertIsInstance(case.run_dir, Path)
            self.assertEqual(tuple(case.fields), ("U",))
            self.assertEqual(case.field("U").field_class, "volVectorField")
            self.assertEqual(case.field("U").boundary_names, ("movingWall", "fixedWalls"))
            self.assertEqual(case.boundary_names, ("fixedWalls", "movingWall"))

    def test_native_directive_uses_declared_toolchain_expansion(self) -> None:
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
            self.assertIn("2", str(value))

    def test_grouped_namespace_naming_is_canonical_and_compatible(self) -> None:
        self.assertIs(fno.OpenFOAM, fno.openfoam)
        self.assertIs(fno.Export, fno.export)
        self.assertIs(fno.Models, fno.models)
        self.assertIs(fno.Runtime, fno.runtime)
        self.assertIn("Case", dir(fno.OpenFOAM))
        self.assertNotIn("Toolchain", dir(fno.OpenFOAM))

    def test_module_shorthand_is_lowered_to_internal_environment(self) -> None:
        case = fno.OpenFOAM.Case(
            case_dir="case",
            run_dir="runs",
            of_cmd="openfoam/2512",
            shell="zsh",
        )
        self.assertEqual(case.of_cmd, "openfoam/2512")
        self.assertEqual(case.shell, "zsh")
        self.assertEqual(case._toolchain.command, "module load openfoam/2512")

    def test_macos_openfoam_command_is_used_as_a_wrapper(self) -> None:
        case = fno.OpenFOAM.Case(
            case_dir="case",
            run_dir="runs",
            of_cmd="openfoam",
            shell="zsh",
        )
        self.assertEqual(case._toolchain.command, "openfoam")
        self.assertTrue(case._toolchain.wrapper)

        from foamnordic._shell import toolchain_shell

        command = toolchain_shell(case._toolchain, "exec pimpleFoam -help")
        self.assertEqual(command[:2], ("zsh", "-lc"))
        self.assertIn("openfoam -c", command[2])
        self.assertIn("exec pimpleFoam -help", command[2])

    def test_macos_wrapper_contains_local_mpi_solver(self) -> None:
        from foamnordic._launch import _solver_command

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
        self.assertEqual(command[:2], ("zsh", "-lc"))
        self.assertIn("openfoam -c", command[2])
        self.assertIn("mpirun -np 6 pimpleFoam", command[2])
        self.assertIn("-parallel", command[2])


if __name__ == "__main__":
    unittest.main()
