# ClosureHost placement and lifecycle

FoamNordic does not expose a database as part of its scientific workflow.
The native service that receives fields, applies bypass rules, evaluates a
closure model, and returns fields is called a **ClosureHost**.

The unified allocation that carries OpenFOAM and its ClosureHosts is called a
**Longship**. Longship is a lifecycle and resource contract, not a data store:
Fjord carries bytes, Harbor owns a peer session, Rune defines messages, and
Longship keeps the solver and model processes in one schedulable job.

Placement, lifecycle, and data-path selection are separate contracts. A
scheduler request must not implicitly select a transport or alter Python
worker behavior.

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

The `foamnordic-longship` executable exposes that supervisor without Python or
an intermediate command shell. One or more `--ready` paths identify regular
marker files or ClosureHost Unix sockets. Arguments following `--host` and
`--solver` are passed directly to their processes:

```bash
foamnordic-longship \
    --ready /tmp/foamnordic-case.sock \
    --host-output closure-host.log \
    --solver-output openfoam.log \
    --host foamnordic_closure_worker \
        unix:///tmp/foamnordic-case.sock model.fnom \
    --solver pimpleFoam -case /path/to/case
```

The supervisor removes stale readiness paths before launch, waits for every
configured host endpoint, starts the solver only afterward, and returns the
failing component's status. Under Slurm the two commands may be `srun` command
arrays; the lifecycle behavior remains identical.
After a successful solver exit, Longship first gives ClosureHost one grace
window to consume protocol shutdown and exit naturally. Only a host that
remains alive receives `SIGTERM` and, after another bounded grace window,
`SIGKILL`. Longship removes every configured readiness path after both process
groups have been reaped, even when the host wrapper itself cannot run cleanup.

External `SIGINT` or `SIGTERM` is converted into a cooperative Longship stop.
The supervisor terminates both component process groups, waits through the
declared grace period, removes readiness markers, and returns status 130.
Python `Run.stop(force=False)` waits on this owned lifecycle. An explicit
`force=True` instead kills the complete local process group or Slurm job and
then reaps the terminal result, which is useful when a notebook must recover
from an unresponsive external runtime.

One resident ClosureHost accepts the declared number of solver connections on
its node and owns one model instance. Every connection retains its global MPI
rank and an independent Harbor, SHM channel, exchange state machine, and
shutdown boundary. Model evaluation is initially serialized inside the host;
this guarantees framework safety without loading one model per rank. A later
batching policy may combine compatible rank requests without changing their
individual exchange identities.

In a Slurm launch, each host writes a distinct readiness marker on a filesystem
visible to the Longship supervisor. `foamnordic_closure_worker` accepts
`--connections N` and `--ready-file PATH`; `{rank}` in the marker path expands
from `SLURM_PROCID`, PMI, PMIx, or Open MPI rank metadata. The marker is created
only after the listener and model are ready and is removed when the worker
exits. The canonical Slurm template passes all markers to
`foamnordic-longship`, which alone owns startup and fail-together behavior.
For a central UCX host, `foamnordic_closure_worker` additionally accepts
`--ucx-host HOST`. This keeps TCP as the advertised control address, creates a
UCP listener on `HOST`, and requires every declared solver connection to
upgrade before inference begins.

When the central host and solver use separate Slurm allocations, Longship runs
`tools/longship/runSlurmClient.sh` as its solver-side process. The adapter
submits one `sbatch` client, records its job identity, waits for accounting,
and cancels the job from its signal/exit cleanup path. Longship therefore still
observes one local process boundary: a host failure terminates the adapter and
cancels the remote client, while a failed client makes the adapter fail and
causes Longship to terminate the host. Slurm submission remains orchestration;
it never enters the field-exchange data path.

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
separately in the [Python API design](../api/design.md). The current source
tree implements only the native placement contract.
