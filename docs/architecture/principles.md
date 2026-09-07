# Design principles
## Stable public surface

FoamNordic exposes cases, operators, field programs, placement, schedulers,
runs, and results. Backend selection stays inside FNOM. Transport and process
layout stay inside the compiled plan.

## Ownership rules

1. The source OpenFOAM case is read-only; every run receives an isolated copy.
2. OpenFOAM owns field storage and equations.
3. ModelHost owns model state and inference workspaces.
4. Fjord owns bytes only while they are in transport.
5. Longship owns child processes until completion, cancellation, or detachment.
6. Results and logs become durable at terminal state.

## Failure rules

- Declarations and artifact contracts fail before launch when possible.
- A request is admitted before tensor bodies are allocated.
- Multiple outputs commit atomically.
- Coupled components fail together.
- Cleanup removes only marker-owned assets.

## Reproducibility

The plan records field bindings, stages, resources, placement, random keys, and
numerics policy. Strict numerics refuses implicit scheme insertion. Convenient
numerics records every actual insertion in the run directory.

## Non-goals

FoamNordic is not a solver, turbulence theory, training framework, experiment
database, or general workflow engine. It does not infer physical meaning from
a model port name or place Python callbacks inside the solver loop.
