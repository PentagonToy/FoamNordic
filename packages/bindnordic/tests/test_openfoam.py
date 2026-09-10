from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from bindnordic.openfoam import read_field


def header(format_name: str, field_class: str, name: str) -> bytes:
    return (
        f"FoamFile {{ format {format_name}; class {field_class}; object {name}; }}\n"
        "dimensions [0 1 -1 0 0 0 0];\n"
    ).encode()


class OpenFOAMTests(unittest.TestCase):
    def test_ascii_vector_metadata_and_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "U"
            path.write_bytes(
                header("ascii", "volVectorField", "U")
                + b"internalField nonuniform List<vector> 2 ((1 2 3) (4 5 6));\n"
                + b"boundaryField { inlet { type fixedValue; } outlet { type zeroGradient; } }\n"
            )
            field = read_field(path)
            self.assertEqual(field.name, "U")
            self.assertEqual(field.boundaries, ("inlet", "outlet"))
            np.testing.assert_allclose(field.values, [[1, 2, 3], [4, 5, 6]])

    def test_uniform_scalar_expands_when_cell_count_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p"
            path.write_bytes(
                header("ascii", "volScalarField", "p")
                + b"internalField uniform 2.5; boundaryField {}\n"
            )
            np.testing.assert_allclose(read_field(path, cell_count=3).values, [2.5] * 3)

    def test_binary_float64_vector(self):
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


if __name__ == "__main__":
    unittest.main()
