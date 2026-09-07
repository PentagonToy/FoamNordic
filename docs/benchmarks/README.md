# Benchmarks and validation

These records use the post-refactor runtime beginning at commit `89cf240`.

| Record | Scope |
|---|---|
| [Transport](transport.md) | UDS, SHM, TCP and UCX across local and HPC systems |
| [Runtime](runtime.md) | Field preparation, model execution, memory and artefacts |
| [OpenFOAM](openfoam.md) | Solver, MPI and end-to-end coupled comparisons |

Pre-refactor records are preserved in [`benchmarks-legacy`](../benchmarks-legacy/README.md).

## Systems

| System | Processor | Memory | Software |
|---|---|---:|---|
| MacBook | Apple M5, 10 cores (4 performance + 6 efficiency), arm64 | 16 GB unified | macOS 26.6.2, Apple Clang 21.0.0 |
| Roihu CPU node | 2 × AMD EPYC 9965, 384 cores, x86_64 | 768 or 1,536 GiB | RHEL 9, GCC 15.2.0 |
| Roihu GPU node | 4 × NVIDIA GH200, 288 Grace cores, aarch64 | 480 GiB LPDDR5 + 384 GiB HBM3 | RHEL 9, JAX 0.11.1, CUDA 13 |

Roihu jobs use only the resources stated with each result. The relevant partitions are:

| Partition | Nodes | CPU and memory | Limit |
|---|---:|---|---|
| `interactive` | 1 | up to 32 CPUs and 64 GiB | 36 h |
| `test` | 1 ~ 2 | up to 384 CPUs and 744 GiB per node | 15 min |
| `small` | 1 | up to 384 CPUs and 1,500 GiB | 72 h |
| `medium` | 1 ~ 6 | full nodes, 384 CPUs and 744 GiB per node | 36 h |
| `gputest` | 1 ~ 2 | 1 ~ 4 GH200 GPUs per node; 217 GiB per GPU | 15 min |

## Method

Microbenchmarks report `median ± maximum deviation` over at least seven warm runs. OpenFOAM comparisons keep the case, allocation, decomposition and output policy identical.

## Current validation

The native gate ran 223 Python tests, 17 native tests and `foamnordic doctor` on the MacBook and a Roihu CPU node. Jobs `1092472` and `1093015` covered two-node TCP and UCX. Interactive job `1095041` covered the 16-rank OpenFOAM comparisons.

Return to the [documentation index](../README.md).
