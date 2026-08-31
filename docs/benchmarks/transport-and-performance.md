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

## 2026-08-21: HPC cross-node TCP (CSC Roihu example)

The server ran in an `interactive` partition and the client ran on another
node in a `small` partition. Node names and job identifiers are redacted.
Each atomic closure exchange sent and
returned one 4 MiB float64 tensor. Thirty-two exchanges completed.

| Data path | Round trips/s | Bidirectional payload MiB/s |
|---|---:|---:|
| TCP | 13.8966 | 111.173 |

Rune metadata, payload endpoints, exchange ordering, solver-time identity,
atomic completion, and clean shutdown were verified. This measurement is the
portable inter-node baseline, not the default closure hot path. Its latency
supports the default Longship policy: place one ClosureHost beside the solver
ranks on every node and use SHM locally.

### Split-allocation confirmation

A later run used an interactive server allocation and a one-node `small`
client allocation. It completed 100 round trips with
one 1 MiB tensor sent and returned per exchange.

| Data path | Round trips/s | Bidirectional payload MiB/s |
|---|---:|---:|
| TCP | 21.5159 | 43.0318 |

The payload size and process placement differ from the measurement above, so
the two rates are validation records rather than a direct performance
comparison. Server, client, atomic exchange validation, and shutdown all
passed.

## 2026-08-21: HPC cross-node UCX (CSC Roihu example)

The server ran in `interactive` and the client ran on another node in `small`.
The UCX 1.20-enabled release build first negotiated
over TCP and then transferred every Rune payload through a UCP stream. It
completed 100 round trips with one 1 MiB float64 tensor sent and returned per
exchange.

| Data path | Round trips/s | Bidirectional payload MiB/s |
|---|---:|---:|
| UCX | 1647.19 | 3294.38 |

`UCX_TLS=rc,ud,sm,self` excluded the UCX TCP transport; Roihu exposed RC and UD
over the mlx5 devices. The run therefore validates an actual fabric-backed UCX
path, including TCP-to-UCX upgrade, Rune metadata, atomic completion, payload
integrity, monotonic exchange identity, and clean shutdown.

The earlier split-allocation TCP confirmation used the same exchange count and
payload size and observed 21.5159 round trips/s and 43.0318 MiB/s. UCX was
about 76.6 times faster in this pair of validation runs. That observed ratio is
not a general performance guarantee and does not replace the OpenFOAM release
gate below.

## 2026-08-28: NACA4412 16-rank a-posteriori closure

An incompressible NACA4412 LES was run to physical time `1` with OpenFOAM
v2512 on the CSC Roihu `small` partition. Both runs used one node, 16 MPI
ranks, the same 89,728-cell mesh, `deltaT=0.0025`, numerical schemes, solution
controls, and output policy. The baseline used the stock `kEqn` model. The
coupled run used `kEqnFjord` with the same `Ck=0.094` and `Ce=1.048`
coefficients and evaluated the closure on every model invocation.

The two allocations had the same shape but ran on different physical nodes,
so this is a development record rather than a controlled hardware benchmark.
All 16 coupled ranks selected rank-local SHM. No TCP or UCX payload path was
used.

| Run | OpenFOAM execution time | OpenFOAM wall time | Longship total |
|---|---:|---:|---:|
| Stock `kEqn` | 47.84 s | 50 s | not applicable |
| FoamNordic `kEqnFjord` | 55.39 s | 71 s | 85 s |
| Coupled / stock | 1.158 | 1.420 | 1.700 |

The solver wall-time ratio is 1.420 and passes the 1.5 release gate below.
Longship spent 14 seconds outside the OpenFOAM solver interval on preparation,
startup, and shutdown. That fixed orchestration interval is reported
separately and is not hidden inside the solver comparison.

### Reconstructed final-field comparison

Both decomposed cases were reconstructed before comparison. The binary
OpenFOAM `internalField` arrays at physical time `1` were decoded as float64
values in mesh order. The relative L2 metric is
`norm(FoamNordic - stock) / norm(stock)`.

| Field | Compared values | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|---:|
| `U` | 269,184 | 5.245674e-02 | 2.267362e-03 |
| `p` | 89,728 | 1.299158e-02 | 1.390809e-02 |
| `k` | 89,728 | 2.225951e-03 | 1.476984e-04 |
| `nut` | 89,728 | 2.722755e-06 | 9.023704e-04 |

The reconstructed fields are not bitwise identical. The table records the
observed accumulated difference between the native OpenFOAM implementation
and the blocking external function closure; it does not declare scientific
equivalence. A case-specific validation tolerance must be agreed before these
numbers can become a numerical pass/fail gate. Both runs completed normally at
the requested final time without closure, transport, or solver failure.

## Release performance gate

The microbenchmarks detect transport regressions. The scientific release gate
is an a-posteriori OpenFOAM comparison under an identical allocation, mesh,
decomposition, timestep sequence, and output policy. Coupled wall time must be
no more than 1.5 times the uncoupled solver wall time for the selected LES
case.
