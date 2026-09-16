from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import foamnordic as fno
from foamnordic.postprocess import dataset as dataset_api


def _field(name: str, values: list[float], *, field_class: str = "volScalarField") -> str:
    body = "\n".join(str(value) for value in values)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class {field_class};
    object {name};
}}
dimensions [0 0 0 0 0 0 0];
internalField nonuniform List<scalar>
{len(values)}
(
{body}
)
;
boundaryField
{{
}}
"""


def _write_field(case_dir: Path, time: str, name: str, values: list[float]) -> None:
    target = case_dir / time
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(_field(name, values), encoding="utf-8")


def _vector_field(name: str, values: list[tuple[float, float, float]]) -> str:
    body = "\n".join(f"({x} {y} {z})" for x, y, z in values)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object {name};
}}
dimensions [0 1 0 0 0 0 0];
internalField nonuniform List<vector>
{len(values)}
(
{body}
)
;
boundaryField
{{
}}
"""


def _write_vector_field(case_dir: Path, time: str, name: str, values: list[tuple[float, float, float]]) -> None:
    target = case_dir / time
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(_vector_field(name, values), encoding="utf-8")


def _write_cube_root_vol_properties(case_dir: Path, coefficient: float = 1.0) -> None:
    target = case_dir / "constant/turbulenceProperties"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"""simulationType LES;
LES
{{
    delta cubeRootVol;
    cubeRootVolCoeffs
    {{
        deltaCoeff {coefficient};
    }}
}}
""",
        encoding="utf-8",
    )


def _addressing(values: list[int]) -> str:
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class labelList;
    object cellProcAddressing;
}}
{len(values)}
(
{' '.join(str(value) for value in values)}
)
;
"""


def _binary_addressing(values: list[int]) -> bytes:
    header = f"""FoamFile
{{
    version 2.0;
    format binary;
    class labelList;
    arch "LSB;label=32;scalar=64";
    object cellProcAddressing;
}}
{len(values)}
(
""".encode("ascii")
    return header + np.asarray(values, dtype="<i4").tobytes() + b"\n)\n;\n"


def test_dataset_parse_writes_one_default_container_and_loads_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [1.0, 2.0])
    _write_field(case_dir, "1", "nut", [3.0, 4.0])

    dataset = fno.Dataset.parse(
        case_dir=case_dir,
        fields=["nut"],
        note="small fixture",
        n_jobs=2,
        progress=True,
        coordinates=False,
    )

    assert dataset.path == case_dir / "dataset/dataset.fdata"
    assert dataset.fields == ("nut",)
    assert dataset.times == (0.0, 1.0)
    assert dataset.note == "small fixture"
    assert "FoamNordic dataset" in capsys.readouterr().out
    table = dataset.load()
    assert table.dtype.names == ("time", "cell", "nut")
    np.testing.assert_array_equal(table["time"], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(table["cell"], [0, 1, 0, 1])
    np.testing.assert_allclose(table["nut"], [1.0, 2.0, 3.0, 4.0])
    reloaded = fno.Dataset.read(path=dataset.path)
    np.testing.assert_array_equal(reloaded.load(times=(1.0, 1.0))["nut"], [3.0, 4.0])
    assert fno.Dataset is fno.Postprocess.dataset


def test_dataset_auto_workers_are_conservative(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [1.0])
    _write_field(case_dir, "1", "nut", [2.0])
    _write_vector_field(case_dir, "0", "C", [(1.0, 2.0, 3.0)])
    monkeypatch.setattr(dataset_api.os, "cpu_count", lambda: 32)

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"], n_jobs=-1)

    assert "n_jobs=-1 resolved to 3" in capsys.readouterr().out
    np.testing.assert_allclose(dataset.load()["nut"], [1.0, 2.0])
    assert dataset_api._dataset_workers(-1, 12, decomposed=False) == 4
    assert dataset_api._dataset_workers(-1, 12, decomposed=True) == 2


def test_dataset_derivation_uses_configured_openfoam_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foamnordic import openfoam

    monkeypatch.setattr(openfoam, "_DEFAULT_OF_CMD", None)
    monkeypatch.setattr(openfoam, "_DEFAULT_SHELL", "bash")
    fno.OpenFOAM.setup(of_cmd="openfoam/2512", shell="bash")
    invocation: list[str] = []

    def run(command: list[str], **_: object) -> None:
        invocation.extend(command)

    monkeypatch.setattr(dataset_api.subprocess, "run", run)
    dataset_api._post_process(tmp_path / "case", "grad(U)", 0.0)

    assert invocation[:2] == ["bash", "-lc"]
    assert "module load openfoam/2512" in invocation[2]
    assert "postProcess" in invocation[2]


def test_dataset_reconstructs_ascii_processor_fields_to_global_cell_order(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    for rank, indices, values in ((0, [2, 0], [30.0, 10.0]), (1, [1], [20.0])):
        processor = case_dir / f"processor{rank}"
        _write_field(processor, "0", "nut", values)
        mesh = processor / "constant/polyMesh"
        mesh.mkdir(parents=True)
        (mesh / "cellProcAddressing").write_text(_addressing(indices), encoding="utf-8")

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)

    np.testing.assert_allclose(dataset.load()["nut"], [10.0, 20.0, 30.0])


def test_dataset_uses_processor_field_when_root_initial_field_is_a_code_stream(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    (case_dir / "0").mkdir(parents=True)
    (case_dir / "0/nut").write_text(
        """FoamFile
{
    format ascii;
    class volScalarField;
    object nut;
}
internalField #codeStream
{
}
;
boundaryField
{
}
""",
        encoding="utf-8",
    )
    processor = case_dir / "processor0"
    _write_field(processor, "0", "nut", [2.0, 1.0])
    mesh = processor / "constant/polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "cellProcAddressing").write_text(_addressing([1, 0]), encoding="utf-8")

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)

    np.testing.assert_allclose(dataset.load()["nut"], [1.0, 2.0])


def test_dataset_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [1.0])
    fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)
    with pytest.raises(FileExistsError):
        fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)


def test_dataset_reconstructs_binary_processor_addressing(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    processor = case_dir / "processor0"
    _write_field(processor, "0", "nut", [2.0, 1.0])
    mesh = processor / "constant/polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "cellProcAddressing").write_bytes(_binary_addressing([1, 0]))

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)

    np.testing.assert_allclose(dataset.load()["nut"], [1.0, 2.0])


def test_dataset_reconstructs_openfoam_binary_processor_addressing(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    processor = case_dir / "processor0"
    _write_field(processor, "0", "nut", [2.0, 1.0])
    mesh = processor / "constant/polyMesh"
    mesh.mkdir(parents=True)
    payload = b"""FoamFile
    version 2.0;
    format binary;
    arch \"LSB;label=32;scalar=64\";
    class labelList;
    object cellProcAddressing;
// header
2
("""
    (mesh / "cellProcAddressing").write_bytes(
        payload + np.asarray([1, 0], dtype="<i4").tobytes() + b"\n\n// ************************************************************************* //\n"
    )

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"], coordinates=False)

    np.testing.assert_allclose(dataset.load()["nut"], [1.0, 2.0])


def test_dataset_stores_static_cell_centres_by_default(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [1.0, 2.0])
    _write_vector_field(case_dir, "0", "C", [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["nut"])

    table = dataset.load()
    assert dataset.coordinates
    assert table.dtype.names == ("time", "cell", "x", "y", "z", "nut")
    np.testing.assert_allclose(table["x"], [1.0, 4.0])
    np.testing.assert_allclose(table["y"], [2.0, 5.0])
    np.testing.assert_allclose(table["z"], [3.0, 6.0])


def test_dataset_derives_missing_gradient_in_a_disposable_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "U", [1.0, 2.0])
    _write_field(case_dir, "0", "nut", [3.0, 4.0])

    def derive(staged_case: Path, function: str, physical_time: float) -> None:
        assert function == "grad(U)"
        assert physical_time == 0.0
        _write_field(staged_case, "0", "grad(U)", [5.0, 6.0])

    monkeypatch.setattr(dataset_api, "_post_process", derive)
    dataset = fno.Dataset.parse(
        case_dir=case_dir,
        fields=["U", "gradU", "nut"],
        coordinates=False,
    )

    assert "Sailor reported: gradU is missing" in capsys.readouterr().out
    assert not list((case_dir / "dataset").glob(".dataset-staging-*"))
    np.testing.assert_allclose(dataset.load()["gradU"], [5.0, 6.0])


def test_dataset_removes_disposable_case_after_derivation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "U", [1.0])

    def fail(case: Path, destination: Path, physical_times: tuple[float, ...]) -> Path:
        destination.mkdir(parents=True)
        raise RuntimeError("post-processing failed")

    monkeypatch.setattr(dataset_api, "_stage_case", fail)

    with pytest.raises(RuntimeError, match="post-processing failed"):
        fno.Dataset.parse(case_dir=case_dir, fields=["gradU"], coordinates=False)

    assert not list((case_dir / "dataset").glob(".dataset-staging-*"))


def test_dataset_staging_enables_configured_initial_field_writes(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [1.0])
    control_dict = case_dir / "system/controlDict"
    control_dict.parent.mkdir(parents=True)
    control_dict.write_text(
        """functions
{
    gradU
    {
        executeControl writeTime;
        writeControl writeTime;
    }
}
""",
        encoding="utf-8",
    )
    original = control_dict.read_text(encoding="utf-8")

    staged = dataset_api._stage_case(case_dir, tmp_path / "staging", [0.0])

    payload = (staged / "system/controlDict").read_text(encoding="utf-8")
    assert "executeControl timeStep;" in payload
    assert "writeControl timeStep;" in payload
    assert control_dict.read_text(encoding="utf-8") == original


def test_dataset_derives_missing_delta_from_cube_root_vol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [3.0, 4.0])
    _write_cube_root_vol_properties(case_dir, coefficient=2.0)

    def derive(staged_case: Path, function: str, physical_time: float) -> None:
        assert function == "writeCellVolumes"
        assert physical_time == 0.0
        _write_field(staged_case, "0", "V", [1.0, 8.0])

    monkeypatch.setattr(dataset_api, "_post_process", derive)
    dataset = fno.Dataset.parse(
        case_dir=case_dir,
        fields=["delta", "nut"],
        coordinates=False,
    )

    assert "delta = 2 * cbrt(V)" in capsys.readouterr().out
    np.testing.assert_allclose(dataset.load()["delta"], [2.0, 4.0])


def test_dataset_prefers_configured_delta_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case"
    _write_field(case_dir, "0", "nut", [3.0])
    control_dict = case_dir / "system/controlDict"
    control_dict.parent.mkdir(parents=True)
    control_dict.write_text(
        """functions
{
    delta
    {
        type exprField;
        field delta;
    }
}
""",
        encoding="utf-8",
    )

    def derive(staged_case: Path, function: str | None, physical_time: float) -> None:
        assert function is None
        _write_field(staged_case, "0", "delta", [1.5])

    monkeypatch.setattr(dataset_api, "_post_process", derive)
    dataset = fno.Dataset.parse(case_dir=case_dir, fields=["delta", "nut"], coordinates=False)

    np.testing.assert_allclose(dataset.load()["delta"], [1.5])
