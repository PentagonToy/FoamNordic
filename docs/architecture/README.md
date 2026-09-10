# Architecture

FoamNordic separates experiment declaration from repeated solver execution. Python prepares an immutable plan and an isolated case; native components own the solver loop, field exchange, model evaluation, and atomic publication.

```text
                             Python API
                                 |
                                 v
                        Longship lifecycle
                     /                       \
                    v                         v
       +------------------+              +------------------+
       | OpenFOAM adapter | <-- Fjord -->|     ModelHost    |
       +------------------+              +------------------+
                  |                                |
                  v                                v
            solver fields                       FNOM model
```

Longship records logs, observations, and results without placing Python callbacks in the solver exchange loop.

## Ownership

| Component | Owns |
| --- | --- |
| Python API | declarations, validation, case preparation |
| Longship | placement, startup, readiness, shutdown, logs |
| OpenFOAM adapter | field access, equation hooks, atomic commit |
| Fjord | bounded numeric tensor transport |
| ModelHost | process residency, packing, BindNordic invocation, output validation |
| BindNordic | FNOM, scalers, model tensors, and backend execution |
| Solver adapter | equations, correction order, physical meaning |

No notebook callback, model object, or scheduler command appears in the solver exchange loop. A solver remains responsible for equations and boundary conditions; FoamNordic evaluates only the declared operator.

## Runtime shape

A local run and a Slurm run use the same plan. Coupled jobs place one resident model worker beside the solver ranks on each node. Same-node payloads normally use shared memory; Unix sockets provide control and fallback transport.

FNOM is the deployed model boundary. It contains tensor contracts, scaling, backend metadata, and the payload.

## Further reading

- [Package boundary](../../packages/README.md): the invariant separating model responsibility from simulation semantics
- [Runtime](runtime.md): request ownership, inference, and failure behavior
- [Transport](transport.md): Rune frames, channels, and admission
- [FNOM](fnom.md): artifact contents, inspection, and trust
- [OpenFOAM](openfoam.md): field and equation integration
- [Lifecycle](lifecycle.md): local and Slurm process ownership
- [Observations](observations.md): bounded monitoring data
