# Developer guide

Use this section to locate the owner of a change before editing source. FoamNordic separates solver semantics, transport, model execution and orchestration so that a new closure or backend does not require changes throughout the repository.

| Change | Start here | Continue with | Verify with |
| --- | --- | --- | --- |
| Python declaration or validation | [`packages/foamnordic/foamnordic/core/`](../../packages/foamnordic/foamnordic/core/) | [`packages/foamnordic/foamnordic/execution/`](../../packages/foamnordic/foamnordic/execution/) | [`test_graph.py`](../../packages/foamnordic/tests/test_graph.py), [`test_plan.py`](../../packages/foamnordic/tests/test_plan.py) |
| Closure or equation boundary | [`src/foamnordic/openfoam/closureHook.H`](../../src/foamnordic/openfoam/closureHook.H) | [`src/foamnordic/openfoam/models/`](../../src/foamnordic/openfoam/models/) | Python contract tests, OpenFOAM build, solver smoke test |
| Generic field transform | [`src/foamnordic/openfoam/fjordExchange.C`](../../src/foamnordic/openfoam/fjordExchange.C) | [`fieldBridge.C`](../../src/foamnordic/openfoam/fieldBridge.C), [`operations/`](../../src/foamnordic/openfoam/operations/) | [`test_graph.py`](../../packages/foamnordic/tests/test_graph.py), [`test_plan.py`](../../packages/foamnordic/tests/test_plan.py), [`tests/openfoam/`](../../tests/openfoam/) |
| Observation | [`src/foamnordic/backend/adapter/observation.hpp`](../../src/foamnordic/backend/adapter/observation.hpp) | [`fieldProgramObservation.C`](../../src/foamnordic/openfoam/fieldProgramObservation.C) | [`tests/backend/adapter/`](../../tests/backend/adapter/), [`test_run.py`](../../packages/foamnordic/tests/test_run.py) |
| Exchange or data plane | [`src/foamnordic/backend/communication/`](../../src/foamnordic/backend/communication/) | [`src/foamnordic/backend/adapter/exchange.cpp`](../../src/foamnordic/backend/adapter/exchange.cpp) | [`tests/backend/communication/`](../../tests/backend/communication/), [`tests/backend/adapter/`](../../tests/backend/adapter/) |
| ModelHost execution | [`src/foamnordic/backend/inference/`](../../src/foamnordic/backend/inference/) | [`packages/bindnordic/`](../../packages/bindnordic/) | [`tests/backend/inference/`](../../tests/backend/inference/), [`packages/bindnordic/tests/native/`](../../packages/bindnordic/tests/native/) |
| FNOM or scaler semantics | [`packages/bindnordic/src/bindnordic/`](../../packages/bindnordic/src/bindnordic/) | [`packages/foamnordic/foamnordic/export.py`](../../packages/foamnordic/foamnordic/export.py) | BindNordic native and Python artefact tests |
| Launch or placement | [`packages/foamnordic/foamnordic/execution/`](../../packages/foamnordic/foamnordic/execution/) | [`src/foamnordic/runtime/`](../../src/foamnordic/runtime/) | [`test_launch.py`](../../packages/foamnordic/tests/test_launch.py), [`test_slurm.py`](../../packages/foamnordic/tests/test_slurm.py), [`tests/runtime/`](../../tests/runtime/) |

## Change route

1. Identify the component that owns the behaviour in the table above.
2. Read its local `README.md` and the corresponding [architecture contract](../architecture/README.md).
3. Change the narrowest owning component; do not mirror numerical or lifecycle logic in another layer.
4. Add a contract or regression test beside the owning component.
5. Follow the [verification guide](verification.md), escalating to OpenFOAM, MPI or Slurm only when the change crosses those boundaries.
6. Update user documentation only when supported behaviour, configuration or output changes; keep implementation notes in source and tests.

## Stable boundaries

OpenFOAM owns equations, mesh state, correction order, boundary conditions and field commits. Fjord and Harbor own bounded transport and exchange identity. ModelHost owns resident execution, admission and backend lifetime. BindNordic owns model tensors, FNOM, scaling and backend evaluation. Longship owns preparation, placement, readiness, failure propagation and teardown.

Do not embed a model backend in an OpenFOAM adapter, move solver equations into ModelHost, apply model scaling in the solver, or make observations block solver progress. A change that appears to require any of these should be treated as an architecture change and reviewed against the [design principles](../architecture/principles.md) first.

## Guides

| Guide | Use it for |
| --- | --- |
| [Field programmes and OpenFOAM](field-programs.md) | Closures, transforms, native algebra, observations and external solver adapters |
| [OpenFOAM source style](openfoam-style.md) | Formatting and reviewing OpenFOAM-linked C++ without changing runtime design |
| [Native runtime](native-runtime.md) | Rune, Fjord, Harbor, ModelHost, Smedja, BindNordic and connectors |
| [Verification](verification.md) | Selecting the smallest sufficient local, OpenFOAM, MPI and Roihu checks |
