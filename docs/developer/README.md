# Developer guide

Use the table to locate the component that owns a change. Read its architecture contract before editing.

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

1. Locate the owning component and read its local README and [architecture contract](../architecture/README.md).
2. Change that component and add relevant behavioural coverage; avoid duplicating numerical or lifecycle logic.
3. Follow the [verification guide](verification.md) and update affected API or usage documentation.

## Stable boundaries

Keep equations in OpenFOAM, model execution in ModelHost and BindNordic, exchange in Fjord and Harbor, and orchestration in Longship. Observations remain non-blocking. The [architecture overview](../architecture/README.md#component-boundaries) define these boundaries.

## Guides

| Guide | Use it for |
| --- | --- |
| [Field programmes and OpenFOAM](field-programs.md) | Closures, transforms, native algebra, observations and external solver adapters |
| [OpenFOAM source style](openfoam-style.md) | Formatting and reviewing OpenFOAM-linked C++ without changing runtime design |
| [Native runtime](native-runtime.md) | Rune, Fjord, Harbor, ModelHost, Smedja, BindNordic and connectors |
| [Verification](verification.md) | Selecting the smallest sufficient local, OpenFOAM, MPI and Roihu checks |
| [Documentation site](documentation.md) | Building and publishing the Zensical site on GitHub Pages |
