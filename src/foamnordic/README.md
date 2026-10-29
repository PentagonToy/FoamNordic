# Native runtime

Native code is grouped by ownership rather than by a particular OpenFOAM solver or model framework.

| Directory | Responsibility |
| --- | --- |
| `runtime/` | Longship value types, placement arithmetic, supervision, and CLI |
| `backend/communication/` | Rune framing and Fjord/Harbor transports |
| `backend/adapter/` | backend-neutral tensor ports, exchanges, plans, and observations |
| `backend/inference/` | resident-worker orchestration and BindNordic kernel integration |
| `backend/connectors/` | native backend adapters to BindNordic kernels |
| `openfoam/` | field bridge, function object, equation adapters, and expressions |

## Dependency direction

The dependency direction is deliberate:

```text
runtime
   ^
communication
   ^        ^
adapter  inference
   ^
OpenFOAM
```

Communication does not know OpenFOAM or model backends. Inference does not own solver fields or model formats; BindNordic owns FNOM parsing, scaling and backend kernels. OpenFOAM adapters do not load models or schedule processes. Templates are build and runtime assets rather than a second implementation.

## OpenFOAM adapters

Equation-level adapters are selected by their native correction boundary, not by broad labels such as laminar, RAS, LES, or combustion. A new adapter must:

1. leave transported equations and thermodynamics with the solver;
2. expose semantic tensor ports through `ClosureHook`;
3. validate all returned fields before committing any of them;
4. correct boundaries and thermodynamics at the solver-owned call site;
5. share the common transport, inference, observation, and lifecycle code;
6. include a solver-integrated numerical acceptance test.

Keep solver-specific field names, table layouts and framework sources in the solver project.
