import struct
import tempfile
from pathlib import Path

import numpy as np
from bindnordic.openfoam import read_field


def header(
    format_name: str,
    field_class: str,
    name: str,
    arch: str | None = None,
) -> bytes:
    arch_entry = "" if arch is None else f' arch "{arch}";'
    return (
        f"FoamFile {{ format {format_name};{arch_entry} class {field_class}; object {name}; }}\n"
        "dimensions [0 1 -1 0 0 0 0];\n"
    ).encode()


def test_ascii_vector_metadata_and_boundaries():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "U"
        path.write_bytes(
            header("ascii", "volVectorField", "U")
            + b"internalField nonuniform List<vector> 2 ((1 2 3) (4 5 6));\n"
            + b"boundaryField { inlet { type fixedValue; } outlet { type zeroGradient; } }\n"
        )
        field = read_field(path)
        assert field.name == "U"
        assert field.boundaries == ("inlet", "outlet")
        np.testing.assert_allclose(field.values, [[1, 2, 3], [4, 5, 6]])


def test_uniform_scalar_expands_when_cell_count_is_explicit():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p"
        path.write_bytes(
            header("ascii", "volScalarField", "p")
            + b"internalField uniform 2.5; boundaryField {}\n"
        )
        np.testing.assert_allclose(read_field(path, cell_count=3).values, [2.5] * 3)


def test_binary_float64_vector():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "U"
        raw = struct.pack("<6d", 1, 2, 3, 4, 5, 6)
        path.write_bytes(
            header("binary", "volVectorField", "U")
            + b"internalField nonuniform List<vector> 2 ("
            + raw
            + b");\nboundaryField {}\n"
        )
        np.testing.assert_allclose(read_field(path).values, [[1, 2, 3], [4, 5, 6]])


def test_binary_payload_may_contain_closing_parenthesis_byte():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p"
        values = np.array(
            [struct.unpack("<d", struct.pack("<Q", 0x3FF0000000000029))[0], 2.0]
        )
        assert b")" in values.astype("<f8").tobytes()
        path.write_bytes(
            header("binary", "volScalarField", "p", "LSB;label=32;scalar=64")
            + b"internalField nonuniform List<scalar> 2 ("
            + values.astype("<f8").tobytes()
            + b");\nboundaryField {}\n"
        )
        np.testing.assert_array_equal(read_field(path).values, values)


def test_binary_payload_braces_do_not_corrupt_boundary_dictionary():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p"
        values = np.array(
            [
                struct.unpack("<d", struct.pack("<Q", 0x3FF000000000007B))[0],
                struct.unpack("<d", struct.pack("<Q", 0x3FF000000000007D))[0],
            ]
        )
        payload = values.astype("<f8").tobytes()
        assert b"{" in payload and b"}" in payload
        path.write_bytes(
            header("binary", "volScalarField", "p", "LSB;label=32;scalar=64")
            + b"internalField nonuniform List<scalar> 2 ("
            + payload
            + b");\nboundaryField { inlet { type zeroGradient; } }\n"
        )
        field = read_field(path)
        np.testing.assert_array_equal(field.values, values)
        assert field.boundaries == ("inlet",)


def test_binary_patch_values_do_not_corrupt_boundary_dictionary():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p"
        internal = np.array([1.0, 2.0], dtype="<f8").tobytes()
        boundary = np.array(
            [
                struct.unpack("<d", struct.pack("<Q", 0x3FF000000000007B))[0],
                struct.unpack("<d", struct.pack("<Q", 0x3FF000000000007D))[0],
            ],
            dtype="<f8",
        ).tobytes()
        path.write_bytes(
            header("binary", "volScalarField", "p", "LSB;label=32;scalar=64")
            + b"internalField nonuniform List<scalar> 2 ("
            + internal
            + b");\nboundaryField {\n"
            + b"inlet { type fixedValue; value nonuniform List<scalar> 2 ("
            + boundary
            + b"); }\noutlet { type zeroGradient; }\n}\n"
        )
        field = read_field(path)
        np.testing.assert_array_equal(field.values, [1.0, 2.0])
        assert field.boundaries == ("inlet", "outlet")


def test_binary_list_marker_inside_numeric_data_is_ignored(tmp_path):
    path = tmp_path / "p"
    payload = b"nonuniform List<scalar> 99 (".ljust(32, b"\0")
    path.write_bytes(
        header("binary", "volScalarField", "p", "LSB;label=32;scalar=64")
        + b"internalField nonuniform List<scalar> 4 ("
        + payload
        + b");\nboundaryField { inlet { type zeroGradient; } }\n"
    )

    field = read_field(path)

    np.testing.assert_array_equal(field.values, np.frombuffer(payload, dtype="<f8"))
    assert field.boundaries == ("inlet",)


def test_binary_arch_selects_float32_and_byte_order():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p"
        values = np.array([1.25, -2.5], dtype=">f4")
        path.write_bytes(
            header("binary", "volScalarField", "p", "MSB;label=32;scalar=32")
            + b"internalField nonuniform List<scalar> 2 (\n"
            + values.tobytes()
            + b");\nboundaryField {}\n"
        )
        np.testing.assert_array_equal(read_field(path).values, [1.25, -2.5])
