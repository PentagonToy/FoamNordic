"""Compile the installed wheel's build kit without a source checkout or OpenFOAM."""
from pathlib import Path
import subprocess
import tempfile

import foamnordic


def main():
    kit = Path(foamnordic.__file__).resolve().parent / "buildkit"
    if not (kit / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Installed wheel has no build kit: {kit}")
    with tempfile.TemporaryDirectory(prefix="foamnordic-wheel-kit-") as temporary:
        root = Path(temporary)
        subprocess.run([
            "cmake", "-S", str(kit), "-B", str(root / "build"),
            "-DCMAKE_BUILD_TYPE=Release", "-DFOAMNORDIC_TESTS=OFF",
            "-DFOAMNORDIC_ONNX_RUNTIME=OFF", "-DFOAMNORDIC_RESIDENT_TOOLS=OFF",
            f"-DCMAKE_INSTALL_PREFIX={root / 'runtime'}",
        ], check=True)
        subprocess.run([
            "cmake", "--build", str(root / "build"), "--parallel", "2",
            "--target", "foamnordic-longship", "foamnordic_adapter", "foamnordic_inference",
        ], check=True)
        subprocess.run([
            "cmake", "--install", str(root / "build"), "--component", "Runtime",
        ], check=True)
        subprocess.run([str(root / "runtime/bin/foamnordic-longship"), "--help"], check=True)
    print("Installed wheel build-kit compilation and Longship smoke test passed")


if __name__ == "__main__":
    main()
