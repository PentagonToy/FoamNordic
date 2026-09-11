# Native runtime
ModelHost executes every declared field program outside the OpenFOAM process. Python prepares the plan and artifacts but is not called during exchanges.

## Request path

```text
solver fields
      |
      v
admitted request
      |
      v
pack -> BindNordic scale/evaluate/inverse scale -> validate
      |
      v
atomic field commit
```

A request is identified by exchange index, solver time index, physical time, rank, tensor contract, and payload size. Input bodies are accepted only after the complete request fits the receive budget.

## Execution contexts

Each context owns a reusable Smedja workspace. Immutable field layouts are compiled once; field addresses, shapes, and active cells are rebound for every request. Active estimates and retained capacity share one context budget. Oversized idle cache is released rather than blocking new work.

Backends declare whether evaluation is concurrent. Non-concurrent backends use a narrow lock around evaluation only; packing and output handling remain request-local.

Backend capabilities distinguish internal compute-thread ownership from safe concurrent evaluation. A table lookup may not own a thread pool yet still use several CPUs by serving independent solver-rank requests concurrently; resource planning must therefore preserve both dimensions instead of assigning CPUs from backend names.

## Outputs

Predictions are matched to declared output names and widths. Scaling, numeric conversion, finite-value checks, and destination alias checks complete before any solver-owned field changes. Multi-output publication is atomic.

## Bypass and empty ranks

A field program may evaluate a selected cell subset while deterministic bypass rules fill the remaining cells. Empty rank-local partitions are valid no-op batches and do not retain packing storage.

## Failure behavior

Malformed input, model failure, invalid output, timeout, transport loss, or cancellation invalidates the exchange. Admission waiters wake, memory credits return, and the coupled supervisor tears down peers within the configured grace period.

## Profiling

`Longship.launch(profile_every=N)` samples the ModelHost and each solver rank once every `N` exchanges. On clean shutdown, `Model telemetry` records workload, memory and model-stage costs. `Solver telemetry` records mean request publication, response wait, request-to-response round trip and atomic field commit times.

Response wait includes queueing, model execution and return transport, so it is not a pure network-latency measurement. `profile_every=0` disables telemetry without clock reads, payload scans or counter updates in the request path.
