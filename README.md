<picture>
  <source media="(prefers-color-scheme: dark)" srcset="others/assets/banner_dark.svg">
  <img src="others/assets/banner_light.svg" alt="FoamNordic" width="960">
</picture>

# FoamNordic

FoamNordic runs OpenFOAM cases and couples solver fields to mathematical functions and machine-learning models. Use `Longship` to launch a baseline case, transform fields during a run, or apply a closure inside a solver equation. OpenFOAM advances the solution without running Python in the solver loop.

[Documentation](https://pentagontoy.github.io/FoamNordic/foamnordic/) · [Installation](docs/installation.md) · [Python API](docs/api/README.md) · [Tutorials](tutorials/README.md)

## OpenFOAM compatibility

| OpenFOAM release | Tested environment |
| --- | --- |
| OpenCFD v2512 | Linux, Apple Silicon |
| OpenCFD v2606 | Apple Silicon |
| Foundation 14 | Linux |

Foundation 14 support is available from this source tree. The published 1.0.12 wheels predate this support.

## Install

Use CPython 3.11 or 3.12 with an existing OpenFOAM installation. For OpenCFD, install the released package.

```console
pip install foamnordic
```

For Foundation 14, use this checkout and follow the [editable installation instructions](docs/installation.md#editable-installation).

Activate the OpenFOAM environment you intend to use, then run `foamnordic build` and `foamnordic doctor`. On a Linux cluster, load the corresponding OpenFOAM module. On Apple Silicon, use the `openfoam` launcher from OpenFOAM.app.

## Quick start

From the repository root, run the supplied lid-driven cavity with OpenCFD v2512. Set `of_cmd` and `shell` to match your OpenFOAM installation.

```python
from pathlib import Path

import foamnordic as fno

case = fno.OpenFOAM.Case(
    case_dir=Path("tutorials/openfoam_tutorials/incompressible/laminar/lidDrivenCavity"),
    run_dir=Path("output"),
    of_cmd="module load openfoam/2512",
    shell="bash",
    application="pimpleFoam",
)
case.initialize(mesh="blockMesh", validate_mesh=True)

run = fno.Longship(case=case).launch()
result = run.stop(timeout=900, progress=True)
result.summary(style="compact")

if result.success:
    result.postprocess.statistics(fields=["U", "p"], time_idx=-1, verbose=True)
```

FoamNordic stages a copy of the source case before running the solver. For Foundation 14, use `application="foamRun"` with a case configured for a Foundation solver module.

## Extend the workflow

- Apply a [field transform](docs/developer/field-programs.md#transform-route) at a solver stage, or use an [equation closure](docs/developer/field-programs.md#closure-route) such as the LES adapters `nutFjord` and `kEqnFjord`.
- Export a model as an [FNOM artefact](docs/api/model-artifact.md) to carry its payload, tensor contract and scaling. The same solver-facing API supports compiled C++, ONNX, Joblib, Equinox and native regular-grid tables.
- Launch on a cluster with [Slurm](docs/api/run-control.md#slurm-launch). Match the OpenFOAM task count to the case rank count.
- Read the [architecture](docs/architecture/README.md), [benchmarks](docs/benchmarks/README.md) and [release notes](CHANGELOG.md) for implementation details, validation results and changes.

## Develop from source

```console
uv sync --all-packages --all-extras
source .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFOAMNORDIC_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest
```

Activate the target OpenFOAM environment before testing its integration. See the [developer guide](docs/developer/README.md) for component boundaries and verification.

FoamNordic is [GPL-3.0](LICENSE) research software developed by **Hanseul Kang** at Aalto University under the supervision of **Shervin Karimkashi**.
