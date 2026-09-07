# Native runtime
ModelHost executes every declared field program outside the OpenFOAM process.
Python prepares the plan and artifacts but is not called during exchanges.

## Request path

```text
solver fields
      |
      v
admitted request
      |
      v
pack -> scale -> evaluate -> inverse scale -> validate
      |
      v
atomic field commit
```

A request is identified by exchange index, solver time index, physical time,
rank, tensor contract, and payload size. Input bodies are accepted only after
the complete request fits the receive budget.

## Execution contexts

Each context owns a reusable Smedja workspace. Immutable field layouts are
compiled once; field addresses, shapes, and active cells are rebound for every
request. Active estimates and retained capacity share one context budget.
Oversized idle cache is released rather than blocking new work.

Backends declare whether evaluation is concurrent. Non-concurrent backends use
a narrow lock around evaluation only; packing and output handling remain
request-local.

## Outputs

Predictions are matched to declared output names and widths. Scaling, numeric
conversion, finite-value checks, and destination alias checks complete before
any solver-owned field changes. Multi-output publication is atomic.

## Bypass and empty ranks

A field program may evaluate a selected cell subset while deterministic bypass
rules fill the remaining cells. Empty rank-local partitions are valid no-op
batches and do not retain packing storage.

## Failure behavior

Malformed input, model failure, invalid output, timeout, transport loss, or
cancellation invalidates the exchange. Admission waiters wake, memory credits
return, and the coupled supervisor tears down peers within the configured
grace period.

## Profiling

Longship.launch(profile_every=N) samples admission, packing, backend waiting,
inference, and output time. Zero disables sampling and adds no clock reads to
the production path.
