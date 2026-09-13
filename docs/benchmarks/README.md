# Benchmarks and validation

These records preserve numerical and performance evidence for the current runtime. They are engineering measurements rather than universal performance claims: each result is valid for the stated model, case, allocation, software and execution path.

## Record index

| Record | Scope | Principal metrics |
| --- | --- | --- |
| [Transport](transport.md) | UDS, shared memory, TCP and UCX | Exchanges per second and payload throughput |
| [Runtime](runtime.md) | Field preparation, model execution, memory and artefact handling | Latency, throughput, load time and resident memory |
| [OpenFOAM](openfoam.md) | Solver, MPI and end-to-end coupled comparisons | Wall time, OpenFOAM time and field error |

## Terminology

| Term | Meaning |
| --- | --- |
| OpenFOAM | Direct execution without FoamNordic |
| OF/FNO | OpenFOAM launched by Longship without an externally coupled field model |
| FNO-coupled | OpenFOAM with a resident FoamNordic closure or transform |
| FNO-native | A symbolic FoamNordic function lowered into the OpenFOAM adapter without ModelHost |
| Cold | Includes the stated first-load, compilation or activation work |
| Warm | Excludes the stated warm-up operation |

## Reporting contract

Microbenchmarks use at least seven warm runs unless a section states another repetition count. A value written as $m \pm d_{\max}$ reports the sample median $m$ and the largest absolute deviation from that median,

$$
d_{\max}=\max_i |x_i-m|.
$$

OpenFOAM comparisons preserve the source case, mesh, decomposition, allocation, time-step policy and output policy between paths. End-to-end wall time includes Longship preparation and shutdown; OpenFOAM execution time is parsed from the solver log. Throughput counts the complete payload stated by the benchmark rather than an inferred hardware bandwidth.

## Systems

| System | Processor | Memory | Software |
| --- | --- | ---: | --- |
| MacBook | Apple M5, 10 cores (4 performance + 6 efficiency), arm64 | 16 GB unified | macOS 26.6.2, Apple Clang 21.0.0 |
| Roihu CPU node | 2 × AMD EPYC 9965, 384 cores, x86_64 | 768 or 1,536 GiB | RHEL 9, GCC 15.2.0 |
| Roihu GPU node | 4 × NVIDIA GH200, 288 Grace cores, aarch64 | 480 GiB LPDDR5 + 384 GiB HBM3 | RHEL 9, JAX 0.11.1, CUDA 13 |

Roihu jobs use only the resources stated with each result.

| Partition | Nodes | CPU and memory | Time limit |
| --- | ---: | --- | ---: |
| `interactive` | 1 | Up to 32 CPUs and 64 GiB | 36 h |
| `test` | 1–2 | Up to 384 CPUs and 744 GiB per node | 15 min |
| `small` | 1 | Up to 384 CPUs and 1,500 GiB | 72 h |
| `medium` | 1–6 | Full nodes, 384 CPUs and 744 GiB per node | 36 h |
| `gputest` | 1–2 | 1–4 GH200 GPUs per node; 217 GiB per GPU | 15 min |

## Validation coverage

| Evidence | Coverage |
| --- | --- |
| Native gate | 223 Python tests, 17 native tests and `foamnordic doctor` on the MacBook and a Roihu CPU node |
| Two-node transport | TCP job `1092472` and UCX job `1093015` |
| Solver-native closures | 16-rank Smagorinsky and kEqn job `1095041` |
| Repeated WALE comparison | Interactive job `1098732` |

Return to the [documentation index](../README.md).
