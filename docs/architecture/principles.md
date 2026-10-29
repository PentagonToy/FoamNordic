# Design principles

Convenience features must preserve solver ownership and numerical semantics.

## Ownership

| Resource | Owner | Invariant |
| --- | --- | --- |
| Source case | User | Read-only; every run receives an isolated copy |
| Mesh, fields and equations | OpenFOAM solver | Never persisted or advanced by ModelHost |
| Model state and workspaces | ModelHost and backend | Resident for the declared process lifetime |
| Bytes in transit | Fjord | Owned only until the atomic exchange completes or fails |
| Child processes | Longship | Owned until completion, cancellation or explicit detachment |
| Terminal files | `Runtime.Result` | Durable after the run reaches a terminal state |

## Execution boundary

> **OpenFOAM computes declared mathematics; ModelHost evaluates exported models and tables.**

Every FNOM model or table runs through ModelHost. BindNordic validates the artefact, applies its fitted scaling and evaluates its backend.

Supported deterministic mathematics declared with `Operator.function` may run inside OpenFOAM. Any calculation that depends on fitted or packaged data remains in ModelHost, including scaling, encoding, PCA and ensemble blending.

Fjord transports typed data, Harbor orders exchanges, ModelHost owns model execution and resources, and the OpenFOAM adapter owns fields, dimensions, equations and boundary treatment.

## Public surface

FoamNordic exposes cases, operators, field programmes, observations, placement, schedulers, runs and results. Backend selection belongs to FNOM, while transport and process layout belong to the compiled plan. Solver-specific physics belongs to an adapter outside the generic runtime.

## Failure semantics

| Boundary | Required behaviour |
| --- | --- |
| Declaration | Reject invalid types, names and incompatible contracts before launch when possible |
| Admission | Reserve the complete request before accepting tensor bodies |
| Evaluation | Keep model failure outside solver-owned field storage |
| Publication | Validate every output and commit all destinations atomically |
| Coupled lifecycle | Terminate peers when one required component fails |
| Cleanup | Remove only paths carrying valid FoamNordic ownership markers |

## Reproducibility

The compiled plan records field bindings, stages, resources, placement, random keys and numerics policy. Strict numerics rejects a missing OpenFOAM scheme. Convenient numerics adds only a required missing scheme to the isolated case and records the exact change.

## Performance objective

The engineering target is an FNO-coupled/OF/FNO wall-time ratio of $1.5\times$–$1.8\times$ for matched cases, meshes, decompositions and allocations. This is a target, not a guarantee. Report model, transport, hardware and measurement method with each benchmark.

## Non-goals

FoamNordic is not an OpenFOAM solver, turbulence theory, training framework, experiment database or general workflow engine. It does not infer physical meaning from model port names, redistribute the solver mesh or insert Python callbacks into the solver loop.
