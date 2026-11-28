<picture>
  <source media="(prefers-color-scheme: dark)" srcset="others/assets/banner_dark.svg">
  <img src="others/assets/banner_light.svg" alt="FoamNordic" width="960">
</picture>

# FoamNordic

Run an OpenFOAM case, inspect its fields, or couple a model to the solver from Python. Start with a baseline case and use the same `Longship` interface when you add a field transform or an equation closure.

[Documentation](https://pentagontoy.github.io/FoamNordic/foamnordic/) · [Installation](docs/installation.md) · [Python API](docs/api/README.md) · [Tutorials](tutorials/README.md)

## OpenFOAM compatibility

| OpenFOAM release | Tested environment |
| --- | --- |
| OpenCFD v2512 | Linux, Apple Silicon |
| OpenCFD v2606 | Apple Silicon |
| Foundation 14 | Linux, Apple Silicon (build and loading) |

Experimental Foundation 14 support is available from FoamNordic 1.0.13.

## Install

Use CPython 3.11 or 3.12 and choose an OpenFOAM installation from the options below.

### Linux

On a cluster with OpenCFD v2512, run these commands in your Python environment. Use your cluster's module name if it differs.

```bash
pip install foamnordic
module load openfoam/2512
foamnordic build
foamnordic doctor
```

### macOS (Apple Silicon)

| Distribution | OpenFOAM for macOS | Launcher |
| --- | --- | --- |
| OpenCFD (ESI) | [OpenFOAM.app](https://github.com/gerlero/openfoam-app) | `openfoam` |
| Foundation | [OpenFOAM-MacOS](https://github.com/PentagonToy/OpenFOAM-MacOS) | `openfoam-14` |

Start the OpenFOAM shell with the launcher above and activate your Python environment. Install FoamNordic 1.0.13 or later, then run `foamnordic build` and `foamnordic doctor` with the selected OpenFOAM environment active.

## Quick start

From the FoamNordic checkout, run the supplied lid-driven cavity and inspect its final velocity and pressure. This example uses the v2512 module above.

```python
import foamnordic as fno

case = fno.OpenFOAM.Case(
    case_dir="tutorials/openfoam_tutorials/incompressible/laminar/lidDrivenCavity",
    run_dir="output",
    of_cmd="openfoam/2512",
    application="pimpleFoam",
)
case.initialize(mesh="blockMesh", validate_mesh=True)

run = fno.Longship(case=case).launch()
result = run.stop(timeout=900, progress=True)
result.summary(style="compact")

if result.success:
    result.postprocess.statistics(fields=["U", "p"], time_idx=-1, verbose=True)
```

FoamNordic writes each run to its own directory under `output/`. With OpenCFD on macOS, use `of_cmd="openfoam"` and `shell="zsh"`. For Foundation 14 on a Slurm cluster, start with the [airFoil2D notebook](tutorials/foamnordic_tutorials/incompressible/foundation_airfoil2d.ipynb).

## Try next

- Transform velocity and track fields during a run in the [field exchange notebook](tutorials/foamnordic_tutorials/incompressible/esi_field_exchange_lid_driven_cavity.ipynb).
- Run [channel395 with WALE](tutorials/foamnordic_tutorials/incompressible/foundation_channel395.ipynb) on four MPI ranks, or [launch your own case with Slurm](docs/api/run-control.md#slurm-launch).
- [Export a model](docs/api/model-artifact.md) and use it as a [solver closure](docs/developer/field-programs.md#closure-route).

## Develop from source

See the [developer guide](docs/developer/README.md), [benchmarks](docs/benchmarks/README.md) and [release notes](CHANGELOG.md).

<details>
<summary>Build and run the development tests</summary>

```console
uv sync --all-packages --all-extras
source .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFOAMNORDIC_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest
```

Activate the target OpenFOAM environment before testing its integration.

</details>

FoamNordic is [GPL-3.0](LICENSE) research software developed by **Hanseul Kang** at Aalto University under the supervision of **Shervin Karimkashi**.
