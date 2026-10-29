from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from foamnordic.execution.run import Result, RunStatus
from foamnordic.postprocess import Case, compare
from foamnordic.postprocess._field import read_field_file

import foamnordic as fno


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


def test_time_idx_and_physical_time_are_mutually_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _scalar(root, "0.1", "p", [1.0])
        post = Case(root)
        with pytest.raises(ValueError, match="mutually exclusive"):
            post.field("p", time_idx=0, physical_time=0.1)


def test_time_idx_is_numeric_and_physical_time_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for time, value in (("0", 0.0), ("0.1", 0.1), ("0.02", 0.02)):
            _scalar(root, time, "p", [value])
        post = Case(root)
        assert post.times == (0.0, 0.02, 0.1)
        np.testing.assert_allclose(post.field("p", time_idx=1), [0.02])
        np.testing.assert_allclose(post.field("p", physical_time=0.1), [0.1])
        np.testing.assert_allclose(post.field("p"), [0.1])


def test_uniform_vector_expands_to_mesh_cells() -> None:
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
        assert values.shape == (4, 3)
        np.testing.assert_allclose(values, [[1.0, 2.0, 3.0]] * 4)


def test_decomposed_fields_concatenate_in_processor_order() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _scalar(root / "processor0", "0.5", "p", [1.0, 2.0])
        _scalar(root / "processor1", "0.5", "p", [3.0])
        post = Case(root)
        assert post.times == (0.5,)
        np.testing.assert_allclose(post.field("p"), [1.0, 2.0, 3.0])


def test_statistics_reduce_vector_cells_and_verbose_uses_presenter(mocker) -> None:
    _mock_patch_1 = mocker.patch(
        "foamnordic.postprocess.case.read_case_field",
        return_value=np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    display = mocker.patch("foamnordic.postprocess.case.display_statistics")
    try:
        with (
            tempfile.TemporaryDirectory() as directory,
        ):
            root = Path(directory)
            (root / "1").mkdir()
            statistics = Case(root).statistics("U", verbose=True)
            assert statistics["min"] == 0.0
            assert statistics["max"] == 5.0
            display.assert_called_once()
    finally:
        mocker.stop(display)
        mocker.stop(_mock_patch_1)


def test_strict_case_comparison_returns_physical_metrics() -> None:
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
        assert round(abs(metrics["mae"] - 2.0 / 3.0), 7) == 0
        assert round(abs(metrics["rmse"] - np.sqrt(2.0 / 3.0)), 7) == 0
        assert metrics["max_abs"] == 1.0
        assert round(abs(metrics["relative_l2"] - np.sqrt(2.0 / 14.0)), 7) == 0


def test_public_namespace_is_compact() -> None:
    assert dir(fno.Postprocess) == ["Case", "compare"]
    assert fno.Postprocess is fno.postprocess


def test_result_postprocess_opens_the_isolated_case() -> None:
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
        assert result.postprocess.path == (root / "case").resolve()
        np.testing.assert_allclose(result.postprocess.field("p"), [2.0])


@pytest.mark.parametrize("decomposed", [False, True])
def test_uniform_field_uses_latest_mesh_owner_before_selected_time(tmp_path, decomposed):
    root = tmp_path / "processor0" if decomposed else tmp_path
    _mesh(root, cells=3)
    mesh = root / "0.5/polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_text('FoamFile { note "nCells:4"; }\n')
    future = root / "2/polyMesh"
    future.mkdir(parents=True)
    (future / "owner").write_text('FoamFile { note "nCells:9"; }\n')
    field = root / "1/p"
    field.parent.mkdir()
    field.write_text(
        "FoamFile { format ascii; class volScalarField; object p; }\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        "internalField uniform 2;\nboundaryField {}\n"
    )
    np.testing.assert_array_equal(Case(tmp_path).field("p", physical_time=1), [2] * 4)


@pytest.mark.parametrize("decomposed", [False, True])
@pytest.mark.parametrize("changed", ["points", "owner", "cellProcAddressing"])
def test_strict_comparison_checks_effective_mesh_instances(tmp_path, decomposed, changed):
    if changed == "cellProcAddressing" and not decomposed:
        pytest.skip("processor addressing only applies to decomposed cases")
    cases = [tmp_path / "reference", tmp_path / "candidate"]
    for case in cases:
        root = case / "processor0" if decomposed else case
        _mesh(root)
        if decomposed:
            (root / "constant/polyMesh/cellProcAddressing").write_text("same")
        _scalar(root, "1", "p", [1, 2, 3])
        mesh = root / "0.5/polyMesh"
        mesh.mkdir(parents=True)
        (mesh / changed).write_text(case.name)
    with pytest.raises(ValueError, match="strict mesh comparison found different"):
        compare(*cases, fields="p", physical_time=1)
    # Array-only comparisons remain an explicit opt-in.
    assert compare(*cases, fields="p", physical_time=1, mesh="shape")["p"]["mae"] == 0


def test_strict_comparison_inherits_mesh_objects_and_ignores_future_updates(tmp_path):
    cases = [tmp_path / "reference", tmp_path / "candidate"]
    for case in cases:
        _mesh(case)
        _scalar(case, "1", "p", [1, 2, 3])
        for time, token in [("0.5", "same"), ("2", case.name)]:
            mesh = case / time / "polyMesh"
            mesh.mkdir(parents=True)
            (mesh / "points").write_text(token)
    assert compare(*cases, fields="p", physical_time=1)["p"]["mae"] == 0
