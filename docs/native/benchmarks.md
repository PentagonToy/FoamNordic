# Native benchmark record

This file records observed development measurements. They are not portable
performance guarantees. Every comparison must retain its payload, machine,
build type, transport direction, and process layout.

## 2026-08-20: macOS arm64 local transport

Release build on an Apple Silicon development machine, 10,000 round trips:

| Channel | Exchanges/s | Payload MiB/s |
|---|---:|---:|
| UDS | 52,850 | 645.1 |
| SHM, same process | 141,646 | 1,729.1 |
| SHM, separate processes | 88,383 | 1,078.9 |

The separate-process result exercises the production POSIX SHM mapping and
macOS UDS wake path rather than a memory-only ring benchmark.

## 2026-08-21: Roihu cross-node TCP

The server ran on `rc5183` in the `interactive` partition and the client ran
on `rc5132` in the `small` partition. Each atomic closure exchange sent and
returned one 4 MiB float64 tensor. Thirty-two exchanges completed.

| Data path | Round trips/s | Bidirectional payload MiB/s |
|---|---:|---:|
| TCP | 13.8966 | 111.173 |

Rune metadata, payload endpoints, exchange ordering, solver-time identity,
atomic completion, and clean shutdown were verified. This measurement is the
portable inter-node baseline, not the default closure hot path. Its latency
supports the default Longship policy: place one ClosureHost beside the solver
ranks on every node and use SHM locally.

## Release performance gate

The microbenchmarks detect transport regressions. The scientific release gate
is an a-posteriori OpenFOAM comparison under an identical allocation, mesh,
decomposition, timestep sequence, and output policy. Coupled wall time must be
no more than 1.5 times the uncoupled solver wall time for the selected LES
case.
