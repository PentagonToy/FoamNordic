# Control and data planes

## Control plane

The Longship Orchestrator owns infrequent operations:

- validate resources and placement;
- stage executables and model artifacts;
- submit, monitor, cancel, and summarize a job;
- distribute endpoint and rank-to-host maps;
- collect logs, exit states, and sparse observations;
- request orderly shutdown.

It carries plan metadata and commands only. It does not calculate tensor
values or execute the declared field-operation graph.

The control plane may use Slurm commands, local process supervision, or TCP.
Its latency does not belong in each closure invocation.

## Data plane

The data plane carries fields and model results:

```text
same node      UDS handshake → SHM bulk transfer
different node, automatic    UCX when verified, otherwise TCP
different node, explicit TCP TCP
different node, explicit UCX UCX or fail before exchange
```

Rune defines the message and atomic completion boundary. Harbor owns the peer
session. Fjord supplies the selected byte channel. Neither Harbor nor Fjord
queries Slurm or decides where a model should run.

## Native closure runtime

FoamNordic assigns each runtime responsibility to one native component:

| Responsibility | FoamNordic owner |
|---|---|
| tensor transfer | Fjord and Rune |
| session and handshake | Harbor |
| transient exchange state | ClosureHost |
| model registry | staged manifests and Longship plan |
| process lifecycle | Longship supervisor |
| durable results | explicit output artifacts |

ClosureHost retains only bounded session state needed for active exchanges. It
is not a user-visible database or a durable result store. Attached inference
therefore avoids durable-store serialization and polling in its hot path.

## Orchestrator observations

Observation traffic is separate from closure traffic. Scalars, convergence
statistics, timings, or occasional snapshots may flow to an orchestrator.
Backpressure or disconnection on the observation channel must not leave a
partially published solver response. Policies may drop, buffer within a fixed
limit, or persist observations, but never block an essential closure unless
the user explicitly requests synchronous observation.

The native adapter implements this as a dedicated observation stream:

```text
read-only native reduction
  → non-waiting byte-bounded ring admission
  → independent relay thread
  → framed observation channel
  → Longship receiver
```

The relay owns neither the closure Harbor nor its SHM ring. Slow rendering can
fill only the observation ring, where the declared drop policy applies. A
disconnected receiver marks the observation stream unhealthy and closes that
stream without changing an in-flight model response. No observation plan
means that none of the ring, relay thread, or framed channel is constructed.
Longship merges configured node-local receivers into a byte-bounded event
stream, assigns a monotonic ingestion index, and preserves source identity and
per-source exchange order. It never waits for all nodes to reach a timestep.
