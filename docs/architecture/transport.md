# Native transport
Fjord carries numeric tensors between OpenFOAM ranks and ModelHost. Rune defines
the binary frame; Harbor applies exchange semantics over a selected channel.

```text
OpenFOAM field view
        |
        v
request -> tensor frames -> complete -> ModelHost
```

## Exchange contract

Before tensor bodies, the solver announces exchange identity, tensor count, and
total payload bytes. ModelHost reserves the complete request, then accepts the
bodies. Completion commits only when count, size, shape, and identity match.

Outputs use the same atomic boundary. Every tensor is validated, floating
outputs must be finite, and publication credit remains held until the receiver
accepts the complete result.

## Channels

| Channel | Intended use |
| --- | --- |
| Unix socket | local control and node-local fallback |
| Shared memory | normal same-node bulk payload path |
| TCP | explicit portable cross-node path |
| UCX | optional high-performance native path |

Capability negotiation selects only a channel supported by both peers. An
explicit request fails if unavailable; it does not silently change transport.

## Memory ownership

Receive buffers, active execution storage, retained workspace capacity, and
output publication each have an explicit owner and bound. Retained storage is
a cache and is released when it would block new work. Cancellation wakes
admission waiters and returns credits.

Each solver rank has an independent exchange. A node-local ModelHost serves
only ranks on its node, keeping shared memory and model state node-local.
