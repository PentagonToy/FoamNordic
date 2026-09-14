# Design principles

These rules define the stable architectural boundary of FoamNordic. Convenience features may reduce declaration work, but they must not move solver ownership into Python or hide a change in numerical semantics.

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

For representative coupled workloads, the engineering objective is an FNO-coupled to OF/FNO end-to-end wall-time ratio between $1.5\times$ and $1.8\times$ under the same case, mesh, decomposition and allocation. This is a design target rather than a runtime guarantee; every benchmark must identify the model, transport, hardware and measurement method.

## Non-goals

FoamNordic is not an OpenFOAM solver, turbulence theory, training framework, experiment database or general workflow engine. It does not infer physical meaning from model port names, redistribute the solver mesh or insert Python callbacks into the solver loop.
