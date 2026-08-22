from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import foamnordic as fno
from foamnordic._run import Result, RunStatus
from foamnordic.postprocess import Case, compare
from foamnordic.postprocess._field import read_field_file


def _mesh(case: Path, token: str = "mesh", cells: int = 3) -> None:
    mesh = case / "constant/polyMesh"
    mesh.mkdir(parents=True, exist_ok=True)
    for name in ("points", "faces", "neighbour"):
        (mesh / name).write_text(f"{token}:{name}\n", encoding="utf-8")
    (mesh / "owner").write_text(
        f'FoamFile {{ note "nPoints:8 nCells:{cells} nFaces:12 nInternalFaces:4"; }}\n'
        f"{token}:owner\n",
        encoding="utf-8",
    )


def _scalar(case: Path, time: str, name: str, values: list[float]) -> None:
    directory = case / time
    directory.mkdir(parents=True, exist_ok=True)
    payload = " ".join(str(value) for value in values)
    (directory / name).write_text(
        "FoamFile { format ascii; class volScalarField; object "
        f"{name}; }}\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        f"internalField nonuniform List<scalar> {len(values)} ({payload});\n"
        "boundaryField {}\n",
        encoding="utf-8",
    )


class PostprocessTests(unittest.TestCase):
    def test_time_idx_and_physical_time_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _scalar(root, "0.1", "p", [1.0])
            post = Case(root)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                post.field("p", time_idx=0, physical_time=0.1)

    def test_time_idx_is_numeric_and_physical_time_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for time, value in (("0", 0.0), ("0.1", 0.1), ("0.02", 0.02)):
                _scalar(root, time, "p", [value])
            post = Case(root)
            self.assertEqual(post.times, (0.0, 0.02, 0.1))
            np.testing.assert_allclose(post.field("p", time_idx=1), [0.02])
            np.testing.assert_allclose(post.field("p", physical_time=0.1), [0.1])
            np.testing.assert_allclose(post.field("p"), [0.1])

    def test_uniform_vector_expands_to_mesh_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _mesh(root, cells=4)
            field = root / "1/U"
            field.parent.mkdir()
            field.write_text(
                "FoamFile { format ascii; class volVectorField; object U; }\n"
                "dimensions [0 1 -1 0 0 0 0];\n"
                "internalField uniform (1 2 3);\n"
                "boundaryField {}\n",
                encoding="utf-8",
            )
            values = read_field_file(field)
            self.assertEqual(values.shape, (4, 3))
            np.testing.assert_allclose(values, [[1.0, 2.0, 3.0]] * 4)

    def test_decomposed_fields_concatenate_in_processor_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _scalar(root / "processor0", "0.5", "p", [1.0, 2.0])
            _scalar(root / "processor1", "0.5", "p", [3.0])
            post = Case(root)
            self.assertEqual(post.times, (0.5,))
            np.testing.assert_allclose(post.field("p"), [1.0, 2.0, 3.0])

    def test_statistics_reduce_vector_cells_and_verbose_uses_presenter(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "foamnordic.postprocess.case.read_case_field",
            return_value=np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        ), patch("foamnordic.postprocess.case.display_statistics") as display:
            root = Path(directory)
            (root / "1").mkdir()
            statistics = Case(root).statistics("U", verbose=True)
            self.assertEqual(statistics["min"], 0.0)
            self.assertEqual(statistics["max"], 5.0)
            display.assert_called_once()

    def test_strict_case_comparison_returns_physical_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            _mesh(reference)
            _mesh(candidate)
            _scalar(reference, "1", "p", [1.0, 2.0, 3.0])
            _scalar(candidate, "1", "p", [2.0, 2.0, 4.0])
            metrics = compare(
                reference,
                candidate,
                fields="p",
                physical_time=1.0,
            )["p"]
            self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
            self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.0 / 3.0))
            self.assertEqual(metrics["max_abs"], 1.0)
            self.assertAlmostEqual(metrics["relative_l2"], np.sqrt(2.0 / 14.0))

    def test_public_namespace_is_compact(self) -> None:
        self.assertEqual(dir(fno.Postprocess), ["Case", "compare"])
        self.assertIs(fno.Postprocess, fno.postprocess)

    def test_result_postprocess_opens_the_isolated_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _scalar(root / "case", "1", "p", [2.0])
            result = Result(
                status=RunStatus.SUCCEEDED,
                exit_code=0,
                elapsed_seconds=1.0,
                work_dir=root,
                longship_log=root / "logs/Sailing.log",
                host_log=root / "logs/Harbor.log",
                solver_log=root / "logs/Sailing.out",
            )
            self.assertEqual(result.postprocess.path, (root / "case").resolve())
            np.testing.assert_allclose(result.postprocess.field("p"), [2.0])


if __name__ == "__main__":
    unittest.main()
