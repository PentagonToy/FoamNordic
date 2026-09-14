# Native runtime

The native runtime is organised by ownership. Follow the data from the solver towards the model and change the first component that owns the required behaviour.

```text
OpenFOAM field view
    -> adapter exchange
    -> Harbor
    -> Rune over SHM, UDS, TCP or UCX
    -> ModelHost
    -> Smedja packing and workspace
    -> BindNordic scaling and backend
    -> atomic response and field commit
```

## Source map

| Component | Responsibility | Primary files | Tests |
| --- | --- | --- | --- |
| Rune | Typed binary frame representation | [`backend/communication/src/rune.cpp`](../../src/foamnordic/backend/communication/src/rune.cpp), [`rune.hpp`](../../src/foamnordic/backend/communication/include/foamnordic/fjord/rune.hpp) | [`test_fjord.cpp`](../../tests/backend/communication/test_fjord.cpp) |
| Channels | Byte transport and transport-specific resources | [`backend/communication/src/`](../../src/foamnordic/backend/communication/src/) | `tests/backend/communication/` |
| Harbor | Request publication, response sequencing and session negotiation | [`harbor.cpp`](../../src/foamnordic/backend/communication/src/harbor.cpp), [`harbor.hpp`](../../src/foamnordic/backend/communication/include/foamnordic/fjord/harbor.hpp) | [`test_fjord.cpp`](../../tests/backend/communication/test_fjord.cpp) |
| Adapter exchange | Tensor contracts, profiling and atomic response application | [`backend/adapter/exchange.cpp`](../../src/foamnordic/backend/adapter/exchange.cpp), [`port.cpp`](../../src/foamnordic/backend/adapter/port.cpp) | `tests/backend/adapter/` |
| ModelHost | Resident programmes, readiness, admission and teardown | [`backend/inference/worker.cpp`](../../src/foamnordic/backend/inference/worker.cpp), [`runner.cpp`](../../src/foamnordic/backend/inference/runner.cpp) | [`test_worker.cpp`](../../tests/backend/inference/test_worker.cpp), [`test_runner.cpp`](../../tests/backend/inference/test_runner.cpp) |
| Smedja | Active-cell packing and reusable execution storage | [`backend/inference/smedja.cpp`](../../src/foamnordic/backend/inference/smedja.cpp) | [`test_exchange.cpp`](../../tests/backend/inference/test_exchange.cpp) |
| Execution pools | Capacity, memory credits and execution contexts | [`backend/inference/execution.cpp`](../../src/foamnordic/backend/inference/execution.cpp) | [`test_admission.cpp`](../../tests/backend/inference/test_admission.cpp), [`test_engine.cpp`](../../tests/backend/inference/test_engine.cpp) |
| BindNordic | FNOM, scalers, tensor contracts and model kernels | [`packages/bindnordic/`](../../packages/bindnordic/) | `packages/bindnordic/tests/` |
| Connector | Adapts one native backend to the packed kernel interface | [`backend/connectors/`](../../src/foamnordic/backend/connectors/) | Backend inference tests and a packaged-runtime test |

## Exchange changes

Change Rune only when the binary frame contract changes. Change a channel only for transport-specific connection, transfer or resource behaviour. Change Harbor for request-response sequencing or capability negotiation. Change the adapter exchange for tensor-view validation, exchange profiling or atomic application semantics.

A protocol change must preserve bounded allocation, explicit byte counts, exchange identity and fail-together behaviour. Add malformed-input coverage as well as a successful round trip, and test shared-memory and stream paths when their framing differs.

## Model execution changes

ModelHost and Smedja may schedule, admit, pack and invoke work, but they do not define a model format or apply solver semantics. FNOM parsing, scaler application and backend-specific evaluation belong in BindNordic. A new native connector adapts a BindNordic kernel to ModelHost and must not include OpenFOAM fields, scheduler choices or transport logic.

The installed `plugin.h` is a low-level connector ABI used by the runtime. It is not yet a complete public custom-backend workflow: FNOM format selection and launch-time plugin discovery remain closed contracts. Do not advertise an external connector as supported until artefact creation, plugin placement, ModelHost loading and packaged validation are available together.

## Scaling contract

The packed backend receives model-space tensors after FNOM input scaling and returns model-space outputs before FNOM output scaling is reversed. OpenFOAM adapters, communication channels and external solver code must not duplicate scaler logic.
