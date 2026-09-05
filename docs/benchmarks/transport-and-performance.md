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

## 2026-09-05: macOS arm64 Smedja dtype packing

Release build, one million cells, three scalar float64 fields packed into one
float32 model tensor, 20 repetitions:

| Path | Elapsed | Source throughput |
|---|---:|---:|
| Specialized fused Smedja pack | 0.0350 s | 13,074 MiB/s |
| Idealized handwritten two-pass cast and interleave | 0.0278 s | 16,455 MiB/s |

The fused path was about 21% slower than the idealized loop, but performs the
real contract lookup, shape validation, active-range validation, metadata
update, and reusable workspace management and avoids three additional
full-field float32 staging arrays. This record is a guardrail, not a claim that
fusion alone accelerates an isolated arithmetic loop.

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
supports the default Longship policy: place one ModelHost beside the solver
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

## 2026-08-28: macOS serial NACA4412 reference-path diagnosis

The same 89,728-cell NACA4412 mesh was run serially for ten timesteps from
`0` to `0.025` with `deltaT=0.0025` on Apple Silicon and OpenFOAM v2606. The
stock and coupled cases used identical initial fields, mesh, numerical
schemes, solution controls, and write intervals. The coupled run selected
rank-local SHM and evaluated the same `Ck=0.094`, `Ce=1.048` function through
the resident Python path.

| Run | OpenFOAM execution time | OpenFOAM wall time | Total wall time |
|---|---:|---:|---:|
| Stock `kEqn` | 9.04 s | 10 s | 10.73 s |
| Python-function `kEqnFjord` | 9.05 s | 10 s | 11 s |
| Native-reference diagnostic | 9.04 s | 10 s | 11 s |

The short local run showed no material solver-time penalty. Longship reported
one second of orchestration outside the solver interval.

### Error onset and growth

The binary float64 `internalField` arrays were compared in mesh order. The
Python-function path differed at the first written timestep and the difference
then accumulated through the coupled PIMPLE/LES trajectory.

| Time | Field | Maximum absolute difference | Relative L2 difference |
|---:|---|---:|---:|
| 0.0025 | `U` | 1.471486e-03 | 2.142881e-05 |
| 0.0025 | `p` | 8.571174e-04 | 3.006978e-05 |
| 0.0025 | `k` | 5.193849e-06 | 7.022485e-07 |
| 0.0025 | `nut` | 2.362574e-09 | 7.090451e-06 |
| 0.0250 | `U` | 3.891371e-02 | 1.587492e-03 |
| 0.0250 | `p` | 1.454831e-02 | 2.519213e-03 |
| 0.0250 | `k` | 2.120177e-03 | 6.159998e-05 |
| 0.0250 | `nut` | 3.971894e-07 | 3.663150e-04 |

To isolate the cause, a temporary diagnostic retained `kEqnFjord`'s native C++
reference values while disabling only the external closure overwrite. The
model class, equation assembly, correction sites, and Longship lifecycle were
otherwise unchanged. `U`, `p`, `k`, and `nut` were bitwise identical to stock
OpenFOAM at both `0.0025` and `0.025`.

This establishes that the `kEqnFjord` C++ equation path can reproduce stock
`kEqn` exactly under the tested build. The observed coupled-field difference
originates in recomputing and overwriting the reference terms through the
Python tensor-arithmetic path, not in SHM payload integrity, model correction
ordering, or OpenFOAM equation assembly. The diagnostic source and installed
runtime were restored after the measurement.

## 2026-08-29: Linux direct mmap of embedded Joblib payloads

A startup microbenchmark on CSC Roihu login nodes used Python 3.12, NumPy
2.5.2, Joblib 1.5.3, and a 32 MiB uncompressed array embedded in one
`FNOBND2` file. The payload began at byte 192, a 64-byte boundary. Its Joblib
array began 272 bytes into the payload and was mapped at absolute FNOM byte
464. The mapped address was 16-byte aligned, the backing filename was the
`.fnom` bundle itself, and prediction and end-to-end payload checks passed.

| Loader | Startup | RSS increase | Temporary payload file | Array backing |
|---|---:|---:|---|---|
| Streamed staging fallback | 10.684 ms | 16 KiB | yes | staged Joblib file |
| Direct absolute-offset mmap | 0.470 ms | 16 KiB | no | original FNOM bundle |

The observed startup ratio was approximately 22.7 in favor of direct mapping.
The two measurements ran on different Roihu login nodes, so the ratio is a
development indication rather than a controlled performance claim. More
importantly, the direct run verified the intended storage semantics on Linux:
`numpy.memmap` referenced the original FNOM file, no payload copy was created,
and the resident model remained correct. The production loader detects its
supported Joblib interface and retains the staged mmap path as a compatibility
fallback.

## 2026-09-05: Roihu direct SHM reads

A Release build ran on one AMD EPYC 9965 Roihu `small` node with eight
allocated CPU cores. Each run completed 10,000 blocking round trips carrying
three 400-cell float64 inputs and one 400-cell float64 output. Nine baseline
and direct-read runs were alternated on the same node. The table reports the
median and retains the existing Rune framing, atomic publication, and wake
protocol in both variants.

| SHM path | Baseline exchanges/s | Direct exchanges/s | Change |
|---|---:|---:|---:|
| Same process | 283,816 | 290,620 | +2.4% |
| Separate processes | 232,409 | 237,277 | +2.1% |

The direct path copies a published slot into the caller's final destination
and keeps a partially consumed slot owned by the reader until its final byte.
It removes the intermediate `std::vector` allocation and second full-payload
copy without changing the public API or wire protocol. The cross-process Linux
SHM correctness test and the complete macOS native test suite passed.

## 2026-09-05: compiled estimator CPU scaling

The same Roihu node evaluated a generated C++ ExtraTrees regressor with 64
trees, maximum depth 16, ten inputs, and one output. Each timing is the median
of repeated calls. The parallel call created its worker threads per invocation;
therefore the small-batch rows include thread creation and joining.

| Rows/call | Serial | 8-thread | Speedup |
|---:|---:|---:|---:|
| 256 | 6.298 ms | 2.621 ms | 2.40x |
| 1,000 | 26.149 ms | 7.341 ms | 3.56x |
| 10,000 | 240.529 ms | 46.810 ms | 5.14x |
| 100,000 | 2,293.838 ms | 153.637 ms | 14.93x |

The 100,000-row path also changes from row-major serial evaluation to the
generated tree-major batch kernel, so its result is not a pure eight-core
scaling number. That path differed from serial accumulation by at most
`1.33e-15` absolute and passed `rtol=1e-12, atol=1e-12`; the smaller paths were
bitwise identical. These results do not justify a persistent worker-pool
dependency: the current generated path already benefits at 256 rows while
remaining deterministic within the documented floating-point tolerance.

## Release performance gate

The microbenchmarks detect transport regressions. The scientific release gate
is an a-posteriori OpenFOAM comparison under an identical allocation, mesh,
decomposition, timestep sequence, and output policy. Coupled wall time must be
no more than 1.5 times the uncoupled solver wall time for the selected LES
case.
