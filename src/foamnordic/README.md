# FoamNordic native source

Native code is grouped by ownership rather than by a particular OpenFOAM
solver or model framework.

| Directory | Responsibility |
| --- | --- |
| `runtime/` | Longship value types, placement arithmetic, supervision, and CLI |
| `backend/communication/` | Rune framing and Fjord/Harbor transports |
| `backend/adapter/` | backend-neutral tensor ports, exchanges, plans, and observations |
| `backend/inference/` | FNOM parsing, preprocessing, model runners, and resident workers |
| `backend/connectors/` | native model connectors only |
| `openfoam/` | field bridge, function object, equation adapters, and expressions |

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

Communication does not know OpenFOAM or model backends. Inference does not
own solver fields. OpenFOAM adapters do not load models or schedule processes.
Templates are build/runtime assets rather than a second implementation.

## OpenFOAM adapters

Equation-level adapters are selected by their native correction boundary, not
by broad labels such as laminar, RAS, LES, or combustion. A new adapter must:

1. leave transported equations and thermodynamics with the solver;
2. expose semantic tensor ports through `ClosureHook`;
3. validate all returned fields before committing any of them;
4. correct boundaries and thermodynamics at the solver-owned call site;
5. share the common transport, inference, observation, and lifecycle code;
6. include a solver-integrated numerical acceptance test.

This is the common pattern retained from surveyed FGM, FPV, flamelet,
progress-variable, and detailed-chemistry OpenFOAM implementations. Their
solver-specific field names, table layouts, and copied framework sources do
not belong in FoamNordic.
