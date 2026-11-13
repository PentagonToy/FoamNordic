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

## Install

Use Python 3.11 or 3.12 and an OpenFOAM runtime. The primary HPC target is Linux with OpenFOAM.com v2512. On Apple Silicon, use OpenFOAM.app.

```console
pip install foamnordic
```

Activate OpenFOAM, then run `foamnordic build` and `foamnordic doctor` in that environment. On macOS, enter it with `openfoam`. See the [installation guide](docs/installation.md) for optional ML dependencies, other installation methods and diagnostics.

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

Pass a `Slurm` scheduler to `Longship` and match the OpenFOAM task count to the case rank count. For coupled runs, Longship starts one resident model worker per solver node. See [Slurm launch](docs/api/run-control.md#slurm-launch) for a complete example and resource settings.

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
