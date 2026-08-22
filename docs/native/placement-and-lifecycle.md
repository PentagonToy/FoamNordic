# ClosureHost placement and lifecycle

FoamNordic does not expose a database as part of its scientific workflow.
The native service that receives fields, applies bypass rules, evaluates a
closure model, and returns fields is called a **ClosureHost**.

The unified allocation that carries OpenFOAM and its ClosureHosts is called a
**Longship**. Longship is a lifecycle and resource contract, not a data store:
Fjord carries bytes, Harbor owns a peer session, Rune defines messages, and
Longship keeps the solver and model processes in one schedulable job.

Placement, lifecycle, and data-path selection are separate contracts. This is
intentional: an earlier design used `share_nodes=True` both as a scheduler
request and as an implicit transport switch. That made a placement detail
control Redis, shared memory, UCX, and Python worker behavior at once.

## Default: automatic

`auto` is the default policy. CPU inference is attached. GPU inference is also
attached when the solver nodes contain the requested GPU; otherwise FoamNordic
selects a central GPU ClosureHost. Placement is therefore derived from actual
resources instead of assuming that every GPU must be remote.

## Attached

The default is `attached` placement. It means:

- the ClosureHost belongs to the same allocation as OpenFOAM;
- one ClosureHost instance is created per OpenFOAM solver node;
- every instance is constrained to the node containing the ranks it serves;
- its startup, failure, cancellation, and shutdown belong to the solver job;
- OpenFOAM does not continue after losing its required ClosureHost.

For one-node jobs this is simply OpenFOAM plus one native ClosureHost on the
same node. For multi-node jobs it avoids gathering all rank fields through one
remote service.

Before submission, `plan_longship` validates that solver ranks divide evenly
across nodes and reserves, per node,

```text
solver tasks per node × solver CPUs per task + ClosureHost CPUs
```

The ClosureHost step starts first. OpenFOAM starts only after every node-local
host is ready. A solver or host failure terminates the other component and the
allocation reports failure. The initial native implementation accepts only
attached placement; central GPU placement must use an explicit later plan and
will never be silently folded into the attached policy.

The canonical Slurm skeleton is
`src/foamnordic/template/slurm/longship.sbatch.in`. It reserves the complete
per-node CPU budget in one allocation, starts separate exclusive host and
solver steps, waits for host readiness, and couples their termination. Template
substitution and submission will belong to the later orchestration API; the
native resource arithmetic remains the authoritative validation layer.

`sail_longship` is the native process supervisor beneath that template. It
starts host and solver commands in separate process groups, removes stale
readiness markers, refuses to start the solver before every requested host is
ready, retains child exit status, and terminates the surviving process group
when either component ends. Graceful termination is bounded and escalates to
`SIGKILL`; therefore a failed model service cannot leave OpenFOAM blocked in a
closure exchange or orphan an `srun` step.

Automatic attached transport selection is:

```text
SHM available                 SHM bulk data + UDS control
SHM unavailable, UDS usable   UDS data and control
otherwise                     TCP data and control
```

SHM and UDS are implementation choices, not placement flags. Failure to create
SHM may therefore fall back to UDS without changing scheduler placement or
scientific exchange semantics.

## Central inference

`central` allows a small number of GPU ClosureHost nodes to serve a larger CPU
OpenFOAM job. It has a separate node identity and may require a separate
scheduler allocation or heterogeneous job component, but it remains part of
the same FoamNordic experiment. Solver completion stops it, host failure fails
the coupled experiment, and cancellation covers both sides.

Automatic central transport selection is:

```text
UCX capability and connectivity verified   UCX data + TCP control
otherwise                                  TCP data and control
```

SHM and UDS are rejected for central hosts. Explicit UCX is strict: it must
fail when unavailable rather than silently becoming TCP. Automatic mode may
fall back before the session begins, but a live exchange never switches its
data path halfway through a batch.

The future orchestration surface that selects these policies is documented
separately in the [Python API design](../python/design.md). The current source
tree implements only the native placement contract.
