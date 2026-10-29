<picture>
  <source media="(prefers-color-scheme: dark)" srcset="others/assets/banner_dark.svg">
  <img src="others/assets/banner_light.svg" alt="FoamNordic" width="960">
</picture>

# FoamNordic

**FOAM Native Orchestration Runtime for Distributed Inference Coupling**

FoamNordic couples OpenFOAM fields to mathematical and machine-learning operators without making Python part of the solver loop. The same `Longship` interface runs an unchanged OpenFOAM case, a field transform, or an equation-level closure supplied by any compatible solver adapter.

[Documentation](https://pentagontoy.github.io/FoamNordic/foamnordic/) · [Installation](https://pentagontoy.github.io/FoamNordic/foamnordic/installation/) · [API](https://pentagontoy.github.io/FoamNordic/foamnordic/api/)

FNOM stores the model payload, tensor contracts, scaling and runtime metadata. BindNordic evaluates it; Fjord exchanges fields over shared memory, Unix sockets, UCX or TCP; Longship manages placement, execution and results.

Python 3.11 and 3.12 are supported. See the [changelog](CHANGELOG.md) for release notes and upgrade guidance.

## Architecture

The diagrams separate workload setup and launch from the field exchange and inference path used during solver execution.

<table>
  <tr>
    <td align="center" valign="bottom"><img src="others/figures/diagram_init.svg" alt="Workload setup and launch" height="440"></td>
    <td align="center" valign="bottom"><img src="others/figures/diagram_runtime.svg" alt="Runtime field exchange and inference" height="440"></td>
  </tr>
  <tr>
    <td align="center"><sub>Figure 1: Setup and launch.</sub></td>
    <td align="center"><sub>Figure 2: Runtime exchange and inference.</sub></td>
  </tr>
</table>

## Requirements

| Component | Requirement |
| --- | --- |
| Python | CPython 3.11 or 3.12 |
| OpenFOAM | A supported installation with its compiler and MPI environment for solver execution |
| Operating system | Linux `x86_64`, Linux `aarch64`, or Apple Silicon |
| ONNX Runtime | Downloaded by `foamnordic build`, or supplied through `FOAMNORDIC_ONNX_RUNTIME_ROOT` |

Linux with OpenFOAM v2512 is the primary HPC target. OpenFOAM.app supplies the native macOS environment.

## Install

Install the PyPI release with pip or uv:

```console
pip install foamnordic
```

Or, in an existing virtual environment:

```console
uv pip install foamnordic
```

Follow the [installation guide](docs/installation.md) for optional environment setup, OpenFOAM activation, build checks and separate main or editable installation instructions. Add `foamnordic[ml]` for CPU model training or `foamnordic[ml-cuda12]` for Linux CUDA 12 training.

Installing the Python package does not install OpenFOAM. Build the integration in the target OpenFOAM environment, then run `foamnordic doctor` there before launching a case.

On macOS with OpenFOAM.app, enter its environment before building:

```console
openfoam
foamnordic build
foamnordic doctor
```

Keep the Python environment containing FoamNordic active inside the OpenFOAM shell.

## Quick start

```python
from pathlib import Path
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="lidDrivenCavity",
    case_dir=Path("cases/lidDrivenCavity"),
    run_dir=Path("output"),
    of_cmd="module load openfoam/2512",
    shell="bash",  # HPC: bash | macOS: zsh
    application="pimpleFoam",
)
case.initialize(ranks=1, mesh="blockMesh", validate_mesh=True)

run = fno.Longship(case=case).launch()
result = run.stop(timeout=900, progress=True)
result.summary(style="compact")

post = result.postprocess
post.statistics(["U", "p"], time_idx=-1, verbose=True)
```

The source case is copied into an isolated run directory. `0.orig/` is accepted when `0/` is absent. With no scheduler, rank one runs directly and multiple ranks use the MPI launcher from the declared OpenFOAM environment.

## Couple a model or function

`Transform` mutates ordinary registered fields at a solver stage. `Closure` enters an equation-specific adapter. FoamNordic ships generic field hooks and two LES adapters (`nutFjord` and `kEqnFjord`) as integration examples; domain solvers own their conservation equations and physical closure semantics.

```python
transform = fno.Transform(
    name="predictVelocity",
    operator=fno.Operator.model("velocity.fnom"),
    inputs={"pressure": fno.Field("p")},
    outputs={"velocity": fno.Field("U")},
    at="time_step_start",
)

run = fno.Longship(case=case, transforms=(transform,)).launch()
```

FNOM selects compiled C++, ONNX, Joblib, Equinox, or a native regular-grid table internally, so model loading does not alter the solver-facing API. `foamnordic inspect` reports whether a backend is data-only or code-bearing. Stochastic functions use `fno.Random.key(seed=42, scope="global")`; deterministic programs need no key.

External solvers can add an equation-level adapter without including the FoamNordic source tree. See the installed-SDK [`externalNutFjord` example](examples/openfoam/externalNutFjord/README.md).

## Slurm resources

OpenFOAM and the node-local model worker have explicit resource declarations. One model worker is placed on every OpenFOAM node only when a coupled program is present.

```python
openfoam_resources = fno.Slurm.openfoam(
    nodes=2,
    ntasks=16,
    cpus_per_task=1,
    mem_per_cpu="2G",
)
model_resources = fno.Slurm.model(
    cpus_per_task=4,
    mem_per_cpu="2G",
)
scheduler = fno.Slurm(
    account="<allocation-account>",
    partition="medium",
    time="00:15:00",
    openfoam=openfoam_resources,
    model=model_resources,
)

case.initialize(ranks=16, mesh=None, validate_mesh=True)
run = fno.Longship(case=case, scheduler=scheduler, transforms=(transform,)).launch()
```

Same-node bulk exchange uses shared memory; multi-node runs place a resident worker beside each node's solver ranks.

## Develop from source

```console
uv sync --all-packages --all-extras
source .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFOAMNORDIC_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest
```

Native tests that compile the OpenFOAM integration require the target OpenFOAM environment to be active.

## Documentation

Read the [FoamNordic section](https://pentagontoy.github.io/FoamNordic/foamnordic/) of the shared Nordic documentation site for guides and API references.

- [Package boundary](packages/README.md)
- [Python API](docs/api/README.md)
- [Architecture](docs/architecture/README.md)
- [Benchmarks and validation](docs/benchmarks/README.md)
- [Tutorials](tutorials/README.md)
- [Development and publishing](others/README.md)

FoamNordic is GPL-3.0 licensed research software. It is developed by **Hanseul Kang** at Aalto University under the supervision of **Shervin Karimkashi**.
