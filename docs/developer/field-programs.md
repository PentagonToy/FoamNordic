# Field programmes and OpenFOAM

A field programme enters OpenFOAM as either a solver-owned `Closure` or a stage-owned `Transform`. Both use the same field, observation and lifecycle infrastructure, but their call sites remain separate because only the solver knows when an equation or correction is physically valid.

## Closure route

| Concern | Owner | Primary files |
| --- | --- | --- |
| User declaration and graph eligibility | Python control plane | [`core/spec.py`](../../packages/foamnordic/foamnordic/core/spec.py), [`core/graph.py`](../../packages/foamnordic/foamnordic/core/graph.py) |
| Dictionary generation | Launch preparation | [`execution/openfoam_rendering.py`](../../packages/foamnordic/foamnordic/execution/openfoam_rendering.py), [`tools/templates/openfoam/`](../../tools/templates/openfoam/) |
| Resident closure invocation | OpenFOAM integration | [`closureHook.H`](../../src/foamnordic/openfoam/closureHook.H), [`closureHook.C`](../../src/foamnordic/openfoam/closureHook.C) |
| Field views and atomic commit | OpenFOAM integration | [`fieldBridge.H`](../../src/foamnordic/openfoam/fieldBridge.H), [`fieldBridge.C`](../../src/foamnordic/openfoam/fieldBridge.C) |
| Algebraic LES correction | Solver adapter | [`models/nutFjord/`](../../src/foamnordic/openfoam/models/nutFjord/) |
| Transported LES state | Solver adapter | [`models/kEqnFjord/`](../../src/foamnordic/openfoam/models/kEqnFjord/) |
| Reaction-rate call site | Solver adapter | [`models/progressRateFjord/`](../../src/foamnordic/openfoam/models/progressRateFjord/) |

When adding an equation-level adapter, begin with the nearest existing model and preserve OpenFOAM ownership of equations and boundary correction. Use `ClosureHook` for the resident request and bind only adapter-owned derived values through `OperationFrame`. The [out-of-tree `externalNutFjord` example](../../examples/openfoam/externalNutFjord/README.md) is the supported reference for an adapter built against the installed SDK.

## Transform route

Generic transforms are rendered as the `foamNordicExchange` function object and execute at one declared solver stage. The Python declaration and stage validation live in `core/spec.py`; rendering lives in `execution/openfoam_rendering.py`; [`fjordExchange.C`](../../src/foamnordic/openfoam/fjordExchange.C) owns runtime sequencing, field snapshots, validation and commit.

Resident transforms pass field views through `AtomicFieldExchange`. Eligible symbolic transforms remain inside OpenFOAM and evaluate through [`operations/frame.C`](../../src/foamnordic/openfoam/operations/frame.C). Native multi-output transforms must evaluate and validate every right-hand side before committing any output.

## Observation and profiling

[`FieldProgramObservation`](../../src/foamnordic/openfoam/fieldProgramObservation.H) converts current OpenFOAM fields into the backend-neutral observation plan. [`NativeAlgebraMonitor`](../../src/foamnordic/openfoam/nativeAlgebraMonitor.H) composes observation, topology tracking and sampled native telemetry without adding those responsibilities to a turbulence model or function object.

An observation is read-only and best-effort. It must not alter field values, determine solver progress or turn a slow notebook consumer into solver backpressure. Profiling remains disabled when `profile_every=0`.

## Required checks

| Change | Minimum evidence |
| --- | --- |
| Declaration, graph or rendering | Focused `test_graph.py` and `test_plan.py` coverage |
| Field expression | `tests/openfoam/test_expression.cpp` and an OpenFOAM integration build |
| Closure output or commit order | Contract test plus a solver smoke test that reads the resulting field |
| Observation or profiling | Adapter test plus a solver smoke test that produces an observation or telemetry record |
| New equation-level adapter | Installed-SDK build test and solver-integrated numerical acceptance test |

Use the complete [verification sequence](verification.md) before handing off a cross-boundary change.
