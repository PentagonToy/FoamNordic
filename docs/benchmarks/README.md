# Benchmarks and validation

This directory is the post-refactor benchmark record for FoamNordic. The measurement series starts at commit `89cf240` and does not silently mix older results with the current runtime, transport, ownership, or artefact lifecycle.

| Record | Scope |
|---|---|
| [Transport](transport.md) | UDS, SHM, TCP and UCX across local and HPC systems |
| [Runtime](runtime.md) | Field preparation, model execution, memory and artefacts |
| [OpenFOAM](openfoam.md) | Solver, MPI and end-to-end coupled comparisons |

The preserved pre-refactor records are in [`benchmarks-legacy`](../benchmarks-legacy/README.md). They are historical context, not the current performance baseline.

## Systems

| System | Processor | Memory | Software |
|---|---|---:|---|
| MacBook | Apple M5, 10 cores (4 performance + 6 efficiency), arm64 | 16 GB unified | macOS 26.6.2, Apple Clang 21.0.0 |
| Roihu CPU node | 2 × AMD EPYC 9965, 384 cores, x86_64 | 768 or 1,536 GiB | RHEL 9, GCC 15.2.0 |
| Roihu GPU node | 4 × NVIDIA GH200, 288 Grace cores, aarch64 | 480 GiB LPDDR5 + 384 GiB HBM3 | RHEL 9, JAX 0.11.1, CUDA 13 |

Roihu jobs use only the resources stated with each result. The relevant partitions are:

| Partition | Nodes | CPU and memory | Limit | Use here |
|---|---:|---|---:|---|
| `interactive` | 1 | up to 32 CPUs and 64 GiB | 36 h | debugging |
| `test` | 1 ~ 2 | up to 384 CPUs and 744 GiB per node | 15 min | short CPU validation and microbenchmarks |
| `small` | 1 | up to 384 CPUs and 1,500 GiB | 72 h | one-node measurements |
| `medium` | 1 ~ 6 | full nodes, 384 CPUs and 744 GiB per node | 36 h | multi-node measurements |
| `gputest` | 1 ~ 2 | 1 ~ 4 GH200 GPUs per node; 217 GiB per GPU | 15 min | GPU validation |

## Method

Correctness must pass before timing. Microbenchmarks report the median and range of at least seven warm runs. OpenFOAM comparisons keep the case, allocation, decomposition and output policy identical, and report solver, model and orchestration time separately.

## Current validation

The post-refactor native gate passed on the MacBook and one Roihu CPU node. Job `1087903` passed 223 Python tests, 17 native tests and `foamnordic doctor`. Multi-node and OpenFOAM performance gates remain pending.

Return to the [documentation index](../README.md).
