# Run-control API

This document describes the Python control plane for a FoamNordic LES
experiment. Compile, isolated case preparation, launch, wait, cancel, result,
and bounded file-backed observation contracts ship today. Sections explicitly
labelled as design targets describe future extensions rather than public API.

## Design character

FoamNordic keeps the Nordic names that describe real architectural boundaries:
Longship owns the coupled workload, ClosureHost evaluates a resident model,
and Fjord carries native messages. The public Python surface otherwise uses
plain scientific names. A user should not need to understand transport or
scheduler internals to run a case, and decorative vocabulary should not hide
resource costs.

Python has four responsibilities:

1. prepare an isolated OpenFOAM workspace;
2. bind a versioned model artifact to field expressions;
3. declare Slurm placement, observations, and lifecycle;
4. inspect bounded observations and durable results.

The orchestration notebook never evaluates or returns a closure field in the
solver loop. `Operator.function()` may run in a separate resident Python
worker; OpenFOAM still exchanges through the same native Fjord boundary.

## Model-build workflow

A training or model-build notebook may use JAX, NumPy, and plotting libraries.
It exports a self-contained artifact before a solver allocation begins.
Constants are frozen into the artifact unless the model contract explicitly
declares them as configurable parameters.

```python
from pathlib import Path

import foamnordic as fno
import jax.numpy as jnp

project = Path("/scratch/<allocation-account>/<user>")
model_dir = project / "FoamNordic/tutorials/incompressible/model"
model_dir.mkdir(parents=True, exist_ok=True)


def keqn(k, velocity_grad, filter_width, *, C_k=0.094, C_e=1.048):
    k_positive = jnp.maximum(k, 0.0)
    sqrt_k = jnp.sqrt(k_positive)
    eddy_viscosity = C_k * filter_width * sqrt_k
    strain = fno.math.dev(2.0 * fno.math.symm(velocity_grad))
    production = eddy_viscosity * fno.math.ddot(velocity_grad, strain)
    dissipation = C_e * sqrt_k / filter_width
    return eddy_viscosity, production, dissipation


artifact = fno.Export.onnx(
    exported_onnx,
    path=model_dir / "kEqnFjord.fnom",
    inputs={
        "k": fno.Tensor.scalar(dtype="float64"),
        "velocity_grad": fno.Tensor.tensor(dtype="float64"),
        "filter_width": fno.Tensor.scalar(dtype="float64"),
    },
    outputs={
        "eddy_viscosity": fno.Tensor.scalar(dtype="float64"),
        "k_production": fno.Tensor.scalar(dtype="float64"),
        "k_dissipation_coeff": fno.Tensor.scalar(dtype="float64"),
    },
    name="kEqnFjord",
    verbose=False,
)
```

The ONNX exporter accepts bytes, a path, or an object exposing
`SerializeToString()`. Joblib and Equinox have matching FNOM exporters:

```python
joblib_artifact = fno.export.joblib(
    voting_regressor,
    path=model_dir / "reactionRate.fnom",
    inputs={"features": fno.Tensor.vector(components=3)},
    outputs={"omega": fno.Tensor.scalar()},
    x_scaler=fitted_input_scaler,
    y_scaler=fitted_output_scaler,
)

equinox_artifact = fno.export.equinox(
    eqx_model,
    path=model_dir / "reactionRateEqx.fnom",
    inputs={"features": fno.Tensor.vector(components=3, dtype="float32")},
    outputs={"omega": fno.Tensor.scalar(dtype="float32")},
    x_scaler=fitted_input_scaler,
    y_scaler=fitted_output_scaler,
)
```

`Longship.launch()` reads the FNOM format and selects the fully native ONNX
worker or managed Joblib/Equinox resident automatically. All formats retain
the same Fjord transport and native packing boundary. Callable JAX-to-ONNX
lowering remains an exporter concern. Every export function is quiet by
default; `verbose=True` displays an Onsaemiro artifact summary.

`x_scaler` and `y_scaler` accept fitted scikit-learn `StandardScaler`,
`MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, or affine
`FunctionTransformer` instances. Export converts
their learned arrays into the native FNOM affine contract. The Python scaler
object is not serialized into the model and is never called during an
OpenFOAM exchange.

The lowering backend must validate its ONNX opset and IR version against the
configured ONNX Runtime. The run-control notebook points to the single
`kEqnFjord.fnom` bundle; ClosureHost reads its embedded ONNX payload directly
without importing JAX, extracting a temporary file, or reconstructing the
training model.

## Case and closure declaration

The source case is immutable input. `initialize()` declares preparation for
the isolated execution copy; it does not generate files in the source case.

```python
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="NACA4412",
    case_dir=project / "FoamNordic/openfoam_tutorials/les/NACA4412",
    run_dir=project / "FoamNordic/tutorials/incompressible/output/NACA4412",
    of_cmd="openfoam/2512",
)
case.initialize(ranks=16, mesh=None, validate_mesh=True)

closure = fno.Closure(
    name="kEqnFjord",
    operator=fno.Operator.model(model_dir / "kEqnFjord.fnom"),
    inputs={
        "k": fno.Field("k"),
        "grad_U": fno.Field.grad("U"),
        "delta": fno.Field.delta(),
    },
    outputs={
        "nut": fno.Field("nut"),
        "kProduction": fno.Field("kProduction"),
        "kDissipationCoeff": fno.Field("kDissipationCoeff"),
    },
    key=fno.Random.key(42),
)
```

`mesh=None` requires an existing `constant/polyMesh` and fails before solver
launch when it is absent. `mesh="blockMesh"` runs `blockMesh` in the copied
run case. `validate_mesh=True` follows either path with `checkMesh`. Parallel
initialization then writes a compatible `decomposeParDict` and decomposes the
prepared mesh for the declared ranks.

NACA4412 is a staged validation target rather than the first software gate.
The native multi-output artifact and a compact `kEqnFjord` cavity run should
pass before its mesh, runtime, and physical settings are introduced. A first
NACA4412 launch should copy the case, retain its existing `0/k` boundary
conditions, and use a deliberately short end time; a longer LES campaign is a
separate user decision.

`Closure` replaces the earlier `Operator.closure` plus `Bridge.couple`
combination. Logical tensor names are bound directly to OpenFOAM expressions
and mutable output fields. The plan compiler rejects missing fields, component
count mismatches, duplicate writers, and unsupported expressions before
submission whenever case metadata is sufficient.

For direct scientific functions, the same declaration infers native component
widths for stored fields, coordinates, `grad(...)`, and LES `delta` before it
packages a resident callable:

```python
def keqn(k, grad_U, delta):
    root_k = fno.Math.sqrt(fno.Math.maximum(k, 0.0))
    nut = 0.094 * delta * root_k
    strain = fno.Math.dev(2.0 * fno.Math.symm(grad_U))
    return {
        "nut": nut,
        "kProduction": nut * fno.Math.ddot(grad_U, strain),
        "kDissipationCoeff": 1.048 * root_k / delta,
    }

closure = fno.Closure(
    name="kEqnFjord",
    operator=fno.Operator.function(keqn),
    inputs={
        "k": fno.Field("k"),
        "grad_U": fno.Field.grad("U"),
        "delta": fno.Field.delta(),
    },
    outputs={
        "nut": fno.Field("nut"),
        "kProduction": fno.Field("kProduction"),
        "kDissipationCoeff": fno.Field("kDissipationCoeff"),
    },
)
```

Closure and OpenFOAM adapter names remain case-sensitive. In particular, the
built-in model is exactly `kEqnFjord`, not `KEqnFjord` or `keqnfjord`.

`fno.Field("p")` binds a stored field, `fno.Field.grad("U")` requests a native
derived expression, `fno.Field.delta()` requests the active LES filter width,
and `fno.Field.coordinate("x")` requests a synthesized cell-centre coordinate.
The older top-level `field()`, `grad()`, and `filter_width()` functions remain
compatible aliases.

A general field mutation uses a separate declaration. Logical model ports can
bind to different OpenFOAM field names, so both `U -> U` perturbations and
models such as `x, y, p -> U` use the same contract:

```python
velocity_model = fno.Operator.model(model_dir / "velocity.fnom")
velocity_transform = fno.Transform(
    name="predictVelocity",
    operator=velocity_model,
    inputs={
        "x_coordinate": fno.field("x"),
        "y_coordinate": fno.field("y"),
        "pressure": fno.field("p"),
    },
    outputs={"predicted_velocity": fno.field("U")},
    at="time_step_start",
    key=fno.Random.key(42),
)
```

`Operator.model()` accepts only `.fnom`; its metadata selects ONNX, Joblib, or
Equinox without exposing backend-specific launch classes.
`Operator.function()` packages a callable and its inferred stored-field
contract inside the marker-owned run directory. Logical input names become
keyword arguments and a returned mapping uses logical output names:

```python
root_key = fno.Random.key(42, scope="global")

def perturb_velocity(velocity, *, key):
    scale_key, noise_key = fno.Random.split(key, 2)
    scale = fno.Random.uniform(scale_key, low=0.995, high=1.005)
    noise = fno.Random.normal(noise_key, shape=velocity.shape, std=1.0e-6)
    return {"updated_velocity": velocity * scale + noise}

velocity_transform = fno.Transform(
    name="perturbVelocity",
    operator=fno.Operator.function(perturb_velocity),
    inputs={"velocity": fno.field("U")},
    outputs={"updated_velocity": fno.field("U")},
    at="time_step_start",
    key=root_key,
)
```

`key`, `exchange_index`, `physical_time`, and `rank` are injected only when
explicitly named by the callable. The key is derived from the declared root,
stable program identity, exchange index, and—under `scope="rank"`—the actual
solver rank. Compatibility `rng` and `seed` injection remains available
temporarily.
`fno.field("U.x")`, `.y`, and `.z` provide mutable component views without
copying an entire vector field through Python.

The public reproducibility input is `fno.Random.Key`, defaulting to
`fno.Random.key(42)`. Equinox materializes it as a JAX key internally.
`scope="global"` gives every rank the same invocation key; `scope="rank"`
derives independent rank-local keys. See the [Random API](random-api.md).

The generic OpenFOAM function-object path guarantees `at="time_step_start"`
and `at="time_step_end"`. `outer_corrector` and `pressure_corrected` are
distinct native stage identifiers—not aliases for a time-step callback—but a
stock application is rejected because it does not publish those internal
PIMPLE boundaries. A combustion or custom PIMPLE solver can implement the
solver-native hook without changing this Python declaration.

## Longship and Slurm

There is no user-visible database, client, or `share_nodes` transport switch.
OpenFOAM and inference resources are declared separately, then combined in one
Slurm allocation. FoamNordic places one model host beside OpenFOAM on each
solver node without treating model CPUs as CPUs for every solver rank.

```python
observations = fno.Observe(
    summaries={
        "nut": ("min", "max"),
        "U": ("min", "max", "l2"),
    },
    interval=100,
)

scheduler = fno.Slurm(
    account="<allocation-account>",
    partition="small",
    time="00:15:00",
    openfoam=fno.Slurm.openfoam(
        nodes=1,
        ntasks=16,
        cpus_per_task=1,
        mem_per_cpu="2G",
    ),
    model=fno.Slurm.model(
        cpus_per_task=8,
        mem_per_cpu="1G",
    ),
)

longship = fno.Longship(
    case=case,
    closures=(closure,),
    observations=(observations,),
    scheduler=scheduler,
    verbose=True,
)

run = longship.launch()
```

For a scheduled run, `launch()` waits until Slurm reports `RUNNING` and then
returns the background handle with its Job ID. The default wait is unbounded;
`launch(start_timeout=900)` returns a still-pending handle after 900 seconds
without cancelling the queued job. `verbose=False` suppresses launch messages
but keeps the same start barrier. When Slurm supplies an estimate,
the submission line appends it as `(est. start: TIMESTAMP)`; the later
`Sailing started at` line remains the authoritative start time.

`Slurm.openfoam()` deliberately follows the names printed in an `#SBATCH`
header: `nodes`, `ntasks`, `cpus_per_task`, and `mem_per_cpu`.
`Slurm.model()` uses the same CPU and memory vocabulary. A model host is always
exactly one ClosureHost task per OpenFOAM node, so its task topology is an
internal invariant rather than another user setting or serialized plan field.
In the example, each model host receives eight CPUs and 8 GiB in total. Model
resources never change the OpenFOAM `ntasks`. With
`Longship(verbose=True)`, construction prints a four-column resource table:
`Resource`, `OpenFOAM`, `Model`, and `Allocation`. The default is quiet.

The shared-memory estimate is

```text
SHM ~= 32 MiB * OpenFOAM ranks * field-program count
```

The 32 MiB factor is two directions times 16 slots times a 1 MiB slot. It is
transport capacity, not model heap. For 16 ranks and one closure this is
approximately 512 MiB. When both OpenFOAM and model memory are explicit,
FoamNordic adds this transport capacity to the Slurm node-memory reservation.
OpenFOAM and the model otherwise share the job's node allocation while
retaining their separately declared CPU and memory budgets.

Submission uses a private environment copy with inherited `SLURM_*`,
`SBATCH_*`, and `SRUN_*` values removed. This prevents a Jupyter or parent batch
allocation from leaking incompatible resource variables into the new job and
does not mutate the notebook's `os.environ`.

For a fair solver baseline, omit closures. The native plan then reserves no
ClosureHost CPU and the launch path leaves the source case's turbulence or
combustion dictionary unchanged:

```python
baseline = fno.Longship(
    case=case,
    scheduler=longship.scheduler,
)
baseline_run = baseline.launch()
```

To run on the current machine instead of submitting a new Slurm job, omit the
scheduler entirely:

```python
local_run = fno.Longship(case=case).launch()
```

This is also the current-allocation path when the Python driver itself was
started inside a Slurm job. Supplying `scheduler=...` always requests a new,
independent allocation.

Ordinary callers do not need to invoke `compile()`; `launch()` performs it
internally and returns a small non-blocking lifecycle handle after the start
barrier.
`longship.compile()` remains an advanced inspection API for validation without
submission. Its immutable plan contains the stable digest, rank-to-host map,
model artifact identities, resource arithmetic, and observation limits.

## Read-only observation

Observation declarations are part of the plan because node-local reductions
must be compiled before launch. The same declaration works for Closure and
general Transform workloads. Transform observations are emitted after the
latest declared solver stage has committed its output fields to OpenFOAM
memory. Consuming the stream is optional, and slow plotting does not block the
solver.

```python
for observation in run.observe(progress=True):
    if observation.exchange_index % 100 == 0:
        table.add_row([
            observation.exchange_index,
            f"{observation.time:.4f}",
            f"{observation.summary['nut'].minimum:.6e}",
            f"{observation.summary['nut'].maximum:.6e}",
            f"{observation.timing.closure_wait:.3f}",
            f"{observation.timing.evaluate:.3f}",
        ])

result = run.stop(force=False)
result.summary(style="compact")
```

No context manager is required. Stopping local observation consumption does
not stop the solver. `launch()` reports the background sailing and returns
immediately. `stop(force=False)` waits for normal completion; without a timeout
it waits indefinitely, while an expired timeout leaves the workload running.
Set `progress=True` on `run.stop()` to read the existing OpenFOAM output
incrementally and show the latest physical `Time` (or steady-solver `Iteration`)
on one transient line. This does not create another log or add work to the
solver, MPI, or Fjord path.
When consuming observations, prefer `run.observe(progress=True)`: it shows the
latest observed physical time and exchange index, and clears the line when the
stream ends. A compact collection therefore needs no hand-written print loop:

```python
records = list(run.observe(progress=True))
result = run.stop(force=False)
```

`stop(force=True)` immediately terminates the complete locally owned process
group or issues Slurm `scancel KILL`, then returns the resulting cancelled
state. There is no separate public `wait()` or `raise_for_status()` step.

An orderly Python or Jupyter shutdown performs best-effort cleanup of owned
workloads. `run.detach()` explicitly allows a job to outlive that kernel.
Abrupt process death and node loss cannot guarantee cleanup.

`result.summary(style="short")` and `"compact"` display Job ID, Name, Status,
Partition, Node, and Elapsed through Onsaemiro. `"long"` and `"expanded"` also
show exit code, work directory, all output paths, and plan digest. Final files
under `logs/` include the scheduler identity. Completed run directories use
`<name>-slurm-<jobid>` or `<name>-local-<timestamp>-<short-hash>`, and the same
identity is appended to Sailing and Harbor filenames. The Sailing log ends
with a compact start/finish and OpenFOAM-versus-orchestration timing summary.
Generated batch and submission scripts are grouped under
`slurm/`. The hidden ownership manifest contains the compiled plan, while
hidden `.foamnordic/` state retains preparation and submission diagnostics and
the scheduler identity needed for cancellation.

There is deliberately no `evaluate()`, `send()`, mutable field view, or client
argument on an observation. Summary reductions occur beside the solver and
cross the observation channel as compact records. The stream reports dropped
records; it never backpressures a closure invocation by default.

## Plotting

Plotting remains ordinary scientific Python. A bounded snapshot owns NumPy
arrays whose lifetime is independent of OpenFOAM memory. Geometry coordinates
are sent with a topology identity and may be reused across later snapshots.

```python
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

snapshot = result.observations.latest("airfoil")
x, z = snapshot.points[:, 0], snapshot.points[:, 2]
velocity = np.linalg.norm(snapshot["U"][:, (0, 2)], axis=1)
nut = snapshot["nut"].ravel()
mesh = tri.Triangulation(x, z)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
for axis, values, label in zip(axes, (velocity, nut), (r"$|\mathbf{U}|$", r"$\nu_t$")):
    contour = axis.tricontourf(mesh, values, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=axis, label=label)
    axis.set(xlabel="x", ylabel="z", aspect="equal")
fig.tight_layout()
```

Full-resolution fields, checkpoints, and VTK products remain filesystem
artifacts. `result.case`, `result.logs`, and `result.artifacts` expose their
durable paths without copying them into the notebook. `result.postprocess`
opens the completed OpenFOAM case for field access and numerical comparison;
see the [Postprocess API](postprocess-api.md). Live snapshots remain for
bounded monitoring rather than publication-quality result analysis.

## API guarantees

FoamNordic validates immutable plans before launch, stages one isolated case
per run, and keeps declared closure execution inside a resident native
workload. `Run` and `Result` retain the durable log, case, and artifact paths;
`run.observe()` carries bounded read-only summaries without participating in
field mutation.
