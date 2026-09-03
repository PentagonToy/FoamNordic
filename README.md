<p align="center">
  <img src="others/icon.png" alt="FoamNordic" width="220">
</p>

# FoamNordic

FoamNordic couples OpenFOAM fields to mathematical and machine-learning
operators without making Python part of the solver loop. The same `Longship`
interface runs an unchanged OpenFOAM case, a field transform, an equation-level
closure, or a progress-variable combustion model.

FNOM is FoamNordic's self-contained model artifact. It stores the executable
payload together with tensor contracts, scaling, runtime requirements, and
compatibility metadata. Fjord moves fields over shared memory, Unix sockets,
UCX, or TCP; Longship owns placement, launch, logs, observations, and results.

Current release: `1.0.4` (Python 3.11–3.12).

## Install

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install foamnordic
```

Wheels include the Python API and native control runtime. Build the small
OpenFOAM integration for each compiler/ABI used on the machine:

```console
# Linux/HPC
module load openfoam/2512
foamnordic build

# Apple Silicon with OpenFOAM.app
openfoam
foamnordic build
```

Check the active Python, native, OpenFOAM, MPI, and ABI-matched runtime without
starting a CFD workload:

```console
foamnordic doctor
```

The runtime is installed below
`~/.local/share/foamnordic/runtime/<platform>/<openfoam-abi>/`. Multiple
OpenFOAM ABIs can coexist and are selected from `Case.of_cmd` at launch.
`FOAMNORDIC_OPENFOAM_LIB` is an advanced override, not a normal setup step.

To use the current source directly:

```console
git clone https://github.com/PentagonToy/FoamNordic.git
cd FoamNordic
python -m pip install -e ./python
module load openfoam/2512
foamnordic build
```

On macOS, replace the module command with `openfoam`. Offline ONNX builds may
set `FOAMNORDIC_ONNX_RUNTIME_ROOT` to an unpacked ONNX Runtime 1.28.0 archive;
workflows that do not need native ONNX can use `foamnordic build --without-onnx`.

## Run an OpenFOAM case

```python
from pathlib import Path
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="lidDrivenCavity",
    case_dir=Path("cases/lidDrivenCavity"),
    run_dir=Path("output"),
    of_cmd="openfoam/2512",
    application="pimpleFoam",
)
case.initialize(ranks=1, mesh="blockMesh", validate_mesh=True)

run = fno.Longship(case=case).launch()
result = run.stop(timeout=900, progress=True)
result.summary(style="compact")

post = result.postprocess
post.statistics(["U", "p"], time_idx=-1, verbose=True)
```

The source case is copied into an isolated run directory and is never modified.
`0.orig/` is accepted when `0/` is absent. With no scheduler, rank one runs
directly and multiple ranks use the MPI launcher from the declared OpenFOAM
environment.

## Couple a model or function

`Transform` mutates ordinary registered fields at a solver stage. `Closure`
enters an equation-specific adapter such as `nutFjord`, `kEqnFjord`, or
`reactionRateFjord`.

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

FNOM selects compiled C++, ONNX, Joblib, or Equinox internally, so model
loading does not alter the solver-facing API. Stochastic functions use
`fno.Random.key(seed=42, scope="global")`; deterministic programs need no key.

## Slurm resources

OpenFOAM and the node-local model worker have explicit resource declarations.
One model worker is placed on every OpenFOAM node only when a coupled program
is present.

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

Same-node bulk exchange uses shared memory; multi-node runs place a resident
worker beside each node's solver ranks. The two-node identity gate has exact
`U` and `p` parity on OpenFOAM v2512.

## Development

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFOAMNORDIC_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m unittest discover -s python/tests -v
```

Supported build targets are Linux `x86_64`, Linux `aarch64`, and Apple Silicon.
Linux/OpenFOAM v2512 is the primary HPC target; OpenFOAM.app provides the
native macOS environment.

## Documentation

- [Python API](docs/api/README.md)
- [Architecture](docs/architecture/README.md)
- [Benchmarks and validation](docs/benchmarks/README.md)
- [Tutorials](tutorials/README.md)
- [Development and publishing](others/README.md)

FoamNordic is GPL-3.0 licensed research software. It is developed by
**Hanseul Kang** at Aalto University under the supervision of
**Shervin Karimkashi**.
