# Provisional run-control API

This document describes the Python direction for a FoamNordic LES experiment.
The compile, isolated case preparation, launch, wait, cancel, and result
contracts now ship; observation streaming and richer preparation options below
remain staged design targets.

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

Python never evaluates or returns a closure field in the solver loop.

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

The current exporter accepts ONNX bytes, an ONNX path, or an object exposing
`SerializeToString()`. Callable JAX-to-ONNX lowering remains an exporter
backend rather than being guessed inside the native packaging step. Every
export function is quiet by default; `verbose=True` displays an Onsaemiro
artifact summary.

The lowering backend must validate its ONNX opset and IR version against the
configured ONNX Runtime. The run-control notebook points to
`kEqnFjord.fnom`; ClosureHost resolves its sibling ONNX payload without
importing JAX or reconstructing the training model.

## Case and closure declaration

The source case is immutable input. `prepare()` materializes an execution
workspace under the output root and restricts cleaning to generated files in
that workspace.

```python
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="NACA4412",
    case_dir=project / "FoamNordic/openfoam_tutorials/les/NACA4412",
    run_dir=project / "FoamNordic/tutorials/incompressible/output/NACA4412",
    of_cmd="openfoam/2512",
    shell="bash",
    ranks=16,
)

closure = fno.Closure(
    name="kEqnFjord",
    artifact=model_dir / "kEqnFjord.fnom",
    inputs={
        "k": fno.field("k"),
        "velocity_grad": fno.grad("U"),
        "filter_width": fno.filter_width(),
    },
    outputs={
        "eddy_viscosity": fno.field("nut"),
        "k_production": fno.field("kProduction"),
        "k_dissipation_coeff": fno.field("kDissipationCoeff"),
    },
)
```

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

A field-only native transformation uses a separate operation declaration. It
does not use `operator=None` as an implicit mode:

```python
velocity_scale = fno.Transform.scale(field="U", factor=1.00005)
```

## Longship and Slurm

There is no user-visible database, Redis port, database memory, client, or
`share_nodes` transport switch. Attached placement means one ClosureHost per
solver node. For 16 one-CPU solver ranks on one node, the allocation also
reserves the declared ClosureHost CPU rather than oversubscribing a solver
rank.

```python
observations = fno.Observe(
    summaries={
        "nut": ("min", "max"),
        "U": ("min", "max", "l2"),
    },
    every=100,
    snapshots={
        "airfoil": fno.Sample.plane(
            normal="y",
            origin=(0.0, 0.0, 0.0),
            fields=("U", "nut"),
            maximum_points=10_000,
        ),
    },
    snapshot_every=1_000,
    retention=fno.Retention.latest(2, maximum="64 MiB"),
)

longship = fno.Longship(
    case=case,
    closures=(closure,),
    observations=(observations,),
    placement=fno.Attached(closure_cpus_per_node=1),
    scheduler=fno.Slurm(
        account="<allocation-account>",
        partition="small",
        time="00:15:00",
        nodes=1,
        ntasks=16,
        cpus_per_task=1,
        mem_per_cpu="1G",
    ),
)

run = longship.launch()
```

For a scheduled run, `launch()` waits until Slurm reports `RUNNING` and then
returns the background handle with its Job ID. The default wait is unbounded;
`launch(start_timeout=900)` returns a still-pending handle after 900 seconds
without cancelling the queued job. `verbose=False` suppresses the two launch
messages but keeps the same start barrier.

`Slurm` deliberately follows the names printed in an `#SBATCH` header:
`nodes`, `ntasks`, `cpus_per_task`, and `mem_per_cpu`. FoamNordic derives the
per-node task count from `ntasks / nodes`; users do not declare the same layout
twice. When a native ClosureHost is attached, its sidecar task is added only to
the compiled allocation and does not change the declared OpenFOAM `ntasks`.
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
and samples must be compiled before launch. Consuming the stream is optional,
and slow plotting does not block the solver.

```python
for observation in run.observe():
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
`stop(force=True)` immediately terminates the complete locally owned process
group or issues Slurm `scancel KILL`, then returns the resulting cancelled
state. There is no separate public `wait()` or `raise_for_status()` step.

An orderly Python or Jupyter shutdown performs best-effort cleanup of owned
workloads. `run.detach()` explicitly allows a job to outlive that kernel.
Abrupt process death and node loss cannot guarantee cleanup.

`result.summary(style="short")` and `"compact"` display Job ID, Name, Status,
Partition, Node, and Elapsed through Onsaemiro. `"long"` and `"expanded"` also
show exit code, work directory, all output paths, and plan digest. Final files
under `logs/` include the scheduler identity, for example
`Sailing_NACA4412_<jobid>.log` and `Sailing_NACA4412_<jobid>.out`; local runs
use `local-<pid>`. Generated batch and submission scripts are grouped under
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
artifacts. `result.case`, `result.logs`, and `result.artifacts` should expose
their durable paths without copying them into the notebook. Live snapshots are
for bounded monitoring; publication-quality post-processing may read the
written OpenFOAM or VTK results after completion.

## Mapping from the historical notebook

| Historical concept | Provisional native API |
| --- | --- |
| `Environment.load()` | `OpenFOAM.Case(of_cmd=..., shell=...)` compiled into launch |
| `Case.initialize(clean=True)` | isolated `Case.prepare(reset="generated")` |
| `Operator.closure(...)` | exported artifact plus `Closure(...)` bindings |
| `Bridge.couple(...)` | closure and observation declarations in the plan |
| `Flow.allocate(...)` | `Slurm(...)` plus explicit attached placement |
| Redis database and port | removed |
| `create_model()` | ClosureHost generated by Longship |
| `connect_client()` | removed |
| `for step ... evaluate(); send()` | native ClosureHost execution |
| monitoring loop | read-only `run.observe()` |
| full `x/y/z/U/nut` transfer | sparse declared snapshot or durable result files |

## Stabilization order

The smallest coherent public release should implement this surface in order:

1. immutable configuration values and plan serialization;
2. case workspace preparation and preflight validation;
3. Slurm rendering and Longship lifecycle handles;
4. ONNX artifact binding and attached ClosureHost placement;
5. read-only summaries, timings, and bounded snapshots;
6. result and artifact discovery;
7. JAX/Joblib export helpers after the native run path is stable.

Until these gates have executable tests, examples must label the API as
provisional and show the equivalent native command where practical.
