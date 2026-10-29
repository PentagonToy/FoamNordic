# FoamNordic

FoamNordic is a native C++ and Python framework for ordinary OpenFOAM and machine-learning field programs, including equation-level closures. It keeps atomic field exchange, packing, and lifecycle in the native runtime, delegates model formats and execution to BindNordic, and exposes a compact declarative API for cases, placement, launch, observations, and results.

> **Note:** The FoamNordic GitHub repository is temporarily private while the software is used in ongoing research and a related publication. The repository will be made public after the research is presented. PyPI distributions remain available in the meantime.

FoamNordic is active research software. Release notes and upgrade guidance are maintained in the [changelog](https://github.com/PentagonToy/FoamNordic/blob/main/CHANGELOG.md).

[Documentation](https://pentagontoy.github.io/FoamNordic/foamnordic/) · [Installation](https://pentagontoy.github.io/FoamNordic/foamnordic/installation/) · [API](https://pentagontoy.github.io/FoamNordic/foamnordic/api/)

## Install

Use Python 3.11 or 3.12. Install the PyPI release with pip:

```console
pip install foamnordic
```

Or use uv in an existing virtual environment:

```console
uv pip install foamnordic
```

Follow the public [installation guide](https://pentagontoy.github.io/FoamNordic/foamnordic/installation/) for optional environment setup, OpenFOAM activation, the integration build and `foamnordic doctor` checks before running a case.

The `ml` extra includes scikit-learn, ONNX, Equinox, Optax and CPU JAX on Linux or macOS. Install `foamnordic[ml-cuda12]` for CUDA 12 JAX on Linux. CUDA selection is explicit rather than inferred from CPU architecture. Install plain `foamnordic` for ordinary OpenFOAM runs, mathematical functions or existing artefacts that do not require those authoring libraries.

Binary wheels carry the native Python control runtime and compact OpenFOAM source build kit. `foamnordic build` compiles it for the currently loaded ABI; each case later selects that runtime from its declared `of_cmd` and `shell`. The build also prepares the native ONNX ModelHost and verified ONNX Runtime 1.28.0. Offline systems can point `FOAMNORDIC_ONNX_RUNTIME_ROOT` at an unpacked 1.28.0 distribution.

On macOS with OpenFOAM.app:

```console
openfoam
foamnordic build
foamnordic doctor
```

Use the Python environment containing FoamNordic inside the OpenFOAM shell. The build calls `wmake` internally.

## First case

```python
from pathlib import Path
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="cavity",
    case_dir=Path("cases/cavity"),
    run_dir="output/cavity",
    of_cmd="module load openfoam/2512",  # macOS OpenFOAM.app: "openfoam"
    shell="bash",
)

# Use an existing cavity case with a blockMeshDict and initial fields.
case.initialize(ranks=1, mesh="blockMesh", validate_mesh=True)
print(case.fields.keys())

run = fno.Longship(case=case).launch()
run.summary()

result = run.stop()
post = result.postprocess
statistics = post.statistics(["U", "p"], time_idx=-1, verbose=True)
```

Paths may be strings, `pathlib.Path`, or text `PathLike` objects. FoamNordic uses `0/` when available and otherwise copies `0.orig/` to `0/` inside the isolated run without changing the source case.

## Platforms

- Linux `x86_64`
- Linux `aarch64`
- macOS Apple Silicon (`arm64`)

Native macOS development uses [OpenFOAM.app](https://github.com/gerlero/openfoam-app). It provides current OpenFOAM releases as Apple Silicon applications and Homebrew packages.

## Learn more

- [Project repository](https://github.com/PentagonToy/FoamNordic)
- [Installation and build guide](https://pentagontoy.github.io/FoamNordic/foamnordic/installation/)
- [Python run-control API](https://pentagontoy.github.io/FoamNordic/foamnordic/api/run-control/)
- [Postprocess API](https://pentagontoy.github.io/FoamNordic/foamnordic/api/postprocess/)
- [Native architecture](https://pentagontoy.github.io/FoamNordic/foamnordic/architecture/)
- [License](https://github.com/PentagonToy/FoamNordic/blob/main/LICENSE)
