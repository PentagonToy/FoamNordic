# FoamNordic architecture

FoamNordic separates declarations, lifecycle, transport, inference, and solver
integration. Python describes a workload and prepares an isolated case; native
components own every repeated field exchange.

```text
Python declaration
    -> immutable execution plan
    -> Longship lifecycle
    -> OpenFOAM adapter <-> Fjord transport <-> ModelHost Smedja
    -> Result and observations
```

## Ownership

| Layer | Owns | Does not own |
| --- | --- | --- |
| Python API | declarations, validation, case preparation | solver loops and repeated field copies |
| Longship | placement, startup, readiness, shutdown, logs | equations and model semantics |
| OpenFOAM adapter | field selection, equation hook, result commit | model loading and scheduling |
| Fjord | versioned tensor transport | OpenFOAM or ML objects |
| ModelHost Smedja | request-local packing, FNOM loading, and inference | solver state and persistent OpenFOAM pointers |
| Solver/model adapter | equations, correction order, thermodynamics | orchestration |

This boundary also applies to domain solvers. A solver owns its transported
variables and equations; FoamNordic evaluates a declared model; a thin solver
adapter validates and commits the result.

## Control and data planes

The control plane may use Python, Slurm, files, and process supervision. The
data plane carries packed numeric tensors through UDS, SHM, TCP, or UCX. A
field exchange never passes through a notebook, scheduler command, database,
or serialized Python object.

Declarations compile before launch. The native plan fixes program order,
ports, tensor layouts, stages, keys, placement, and observation schedules.
Runtime code follows that plan and cannot invent fields or reorder field programs.

## Placement

Node-local inference is the public placement: one ModelHost is started per
solver node and serves the ranks on that node. This keeps model payloads and
shared memory local while preserving one fail-together Slurm allocation. Local
runs use the same lifecycle without a scheduler. TCP and UCX remain native
transport capabilities and validation tools, not a second public placement API.

## Model execution

FNOM is the sole deployed model contract. It contains the tensor contract,
preprocessing, backend metadata, and payload required at runtime. ONNX and
compiled estimators execute natively; Joblib and Equinox use a managed Python
resident. Backend selection remains inside FNOM, so `Operator.model(...)` is
backend-neutral.

## Documents

| Document | Contents |
| --- | --- |
| [Native runtime](runtime.md) | Exchanges, resident execution, bypass, and failures |
| [Native transport](transport.md) | Rune, Fjord, Harbor, UDS, SHM, TCP, UCX, and MPI layout |
| [FNOM](fnom.md) | Artifact encoding, compatibility, preprocessing, and trust |
| [OpenFOAM integration](openfoam.md) | Field bridge, equation adapters, and native build contract |
| [Lifecycle](lifecycle.md) | Node-local placement, resource plans, startup, and shutdown |
| [Observations](observations.md) | Monitoring, reduction, retention, and backpressure |

Measured behavior belongs in [benchmarks and validation](../benchmarks/README.md),
not in architecture promises.
