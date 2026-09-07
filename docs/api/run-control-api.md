# Run control API
FoamNordic describes a workload with immutable declarations and returns a non-blocking Run. The source case is copied before any preparation command.

## Case

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
```

`of_cmd` may be an environment module shorthand or an explicit shell command. `mesh=None` uses an existing mesh. If `0/` is absent, `0.orig/` is copied into the isolated case.

## Models and field programs

```python
transform = fno.Transform(
    name="predictVelocity",
    operator=fno.Operator.model("velocity.fnom"),
    inputs={"pressure": fno.Field("p")},
    outputs={"velocity": fno.Field("U")},
    at="time_step_start",
)

closure = fno.Closure(
    name="nutFjord",
    operator=fno.Operator.model("eddy-viscosity.fnom"),
    inputs={
        "velocity_grad": fno.Field.grad("U"),
        "delta": fno.Field.delta(),
    },
    outputs={"nut": fno.Field("nut")},
)
```

`Transform` is a generic stored-field operation. `Closure` requires a solver-side equation adapter. FNOM selects ONNX, compiled, Joblib, or Equinox without changing this interface.

Pure mathematical functions can use `Operator.function`; logical inputs become keyword arguments and returned mapping keys are logical outputs.

## Local launch

```python
longship = fno.Longship(
    case=case,
    transforms=(transform,),
    numerics="strict",  # optional; default is convenient
)
run = longship.launch(start_timeout=900, profile_every=0)
result = run.stop(timeout=3600, progress=True)
result.summary(style="compact")
```

With no scheduler, rank one runs directly and multiple ranks use the MPI launcher associated with the case environment.

## Slurm launch

```python
scheduler = fno.Slurm(
    account="project_0000000",
    partition="small",
    time="00:30:00",
    openfoam=fno.Slurm.openfoam(
        nodes=1, ntasks=16, cpus_per_task=1, mem_per_cpu="2G"
    ),
    model=fno.Slurm.model(cpus_per_task=4, mem_per_cpu="2G"),
)

case.initialize(ranks=16, mesh=None, validate_mesh=True)
run = fno.Longship(
    case=case, scheduler=scheduler, closures=(closure,)
).launch(start_timeout=900)
```

Solver and model resources are separate. Coupled jobs place a resident model worker on every solver node and fail together inside one allocation.

## Observations

`Observe` requests bounded native summaries at a declared interval. Observation records are monitoring data and never gate solver progress. Stored fields and full scientific comparison belong to `Result.postprocess`.

## Run ownership

- `run.status` reads current state.
- `run.stop()` waits for normal completion.
- `run.stop(force=True)` requests bounded cancellation.
- `run.detach()` allows the workload to outlive Python ownership.
- `result.artifacts` exposes the isolated case, logs, observations, and Slurm files.

Closing an owning Python process cancels its active workload after the orphan timeout. A detached run must be managed through its scheduler or supervisor.

## Reproducibility

The run marker stores the compiled plan and digest. It includes resources, placement, stages, bindings, random keys, and numerics policy. Convenient numerics records actual dictionary insertions in the run directory.
