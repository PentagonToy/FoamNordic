# Execution topologies

## A. Attached node-local inference

This is the default and the performance target.

```text
Longship allocation
├── solver node 0
│   ├── OpenFOAM ranks 0 ... n
│   └── ClosureHost 0
│       └── native or resident model
├── solver node 1
│   ├── OpenFOAM ranks n+1 ... m
│   └── ClosureHost 1
│       └── native or resident model
└── Longship supervisor
```

Each rank exchanges only with the ClosureHost on its own node. UDS establishes
the session and POSIX SHM carries bulk tensors. Linux uses futex wakeups; macOS
retains UDS as its blocking wake channel. No complete field is gathered through
rank zero, a notebook, or another node.

The model artifact may be copied once per node, but its parameters are loaded
once by the node host rather than once per rank. A host may batch compatible
rank requests only when doing so preserves each blocking closure invocation,
exchange index, solver time, and output identity.

## B. External Longship orchestrator with attached inference

Papermill, a notebook, or a future FoamNordic service may live outside the
solver allocation.

```text
Longship Orchestrator
    │ submit / cancel / status / logs / sparse observations
    ▼
Longship allocation
├── solver node 0: OpenFOAM ranks ↔ local ClosureHost
├── solver node 1: OpenFOAM ranks ↔ local ClosureHost
└── native supervisor
```

The orchestrator remains a control-plane peer. It does not receive every
velocity, pressure, progress-variable, or reaction-rate tensor. Disconnecting
an interactive notebook must not corrupt an active atomic exchange. Depending
on policy, the submitted job may continue autonomously or be cancelled as one
Longship, but its node-local data path does not change.

Sparse observations and summaries may be published to the orchestrator at an
explicit cadence. A full-field snapshot is an observation artifact, not the
closure response path.

The observation channel and retention rules are defined separately in
[Observations and retention](observations-and-retention.md).

## C. Central accelerator inference

This topology is explicit and intended for a model that materially benefits
from a GPU unavailable on solver nodes.

```text
Longship Orchestrator
    │ control
    ▼
GPU ClosureHost
    ⇅ UCX preferred, TCP fallback only in automatic mode
CPU node 0 gateway ⇄ SHM ⇄ OpenFOAM ranks
CPU node 1 gateway ⇄ SHM ⇄ OpenFOAM ranks
CPU node 2 gateway ⇄ SHM ⇄ OpenFOAM ranks
```

Every CPU node retains a local gateway so OpenFOAM ranks never create a dense
all-to-all connection set to the GPU service. The gateway validates and packs
rank-local exchanges, sends inter-node batches, and scatters verified outputs
back through SHM. Exchange identity remains per solver invocation even when
transport batching is used.

UCX is valuable here because bulk fields genuinely cross nodes. TCP remains a
portable correctness path. An explicit `ucx` request fails if UCX cannot be
negotiated; only `automatic` may fall back to TCP before the first exchange.

## D. Remote interactive field callback

Routing every field through a notebook to execute code such as
`U * 1.00005` is supported only as an explicit diagnostic topology. It is not a
production default: notebook scheduling, Python copies, network latency, and
kernel failure enter the solver's blocking path.

The preferred form is to declare the operation before launch. FoamNordic then
installs it as a node-local native operation plan. If the operation requires
Python, a Python executor is attached beside each node-local ClosureHost and
loads the callback once. The notebook submits the declaration but does not
execute the timestep loop.

The complete loop-free contract is described in
[Declarative execution plans](declarative-plans.md).

## Invariants shared by every topology

- every closure invocation receives a fresh monotonic exchange index;
- repeated PIMPLE/PISO outer-corrector calls at one physical time are distinct;
- a response is applied only after every declared output and completion record
  has been validated;
- solver and model failures are coupled by Longship;
- changing placement never changes the scientific field contract;
- a live session never silently changes its data path.
