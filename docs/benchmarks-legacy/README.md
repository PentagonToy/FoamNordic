# Benchmarks and validation — legacy record

These measurements predate the runtime and ownership refactor completed at commit `89cf240`. They are preserved unchanged as historical evidence and are not the current performance baseline. New measurements live in [`docs/benchmarks`](../benchmarks/README.md).

These documents collect measured evidence and remaining acceptance gates. They are separated from implementation descriptions so observed results are not confused with architectural promises.

| Record | Contents |
|---|---|
| [Transport and performance](transport-performance.md) | macOS and Roihu transport measurements plus the OpenFOAM performance gate |
| [HPC transport](hpc-transport.md) | Implemented data planes, UCX environment, and portable site checks |
| [OpenFOAM cases](openfoam-cases.md) | Laminar, RAS, LES, and copied-case compatibility evidence |
| [Closure validation](closure-validation.md) | Mathematical parity and learned-closure accuracy and timing |
| [Compiled estimator](compiled-estimator.md) | Compiled C++ versus Joblib startup, memory, parity, and batch-size crossover |
| [Acceptance status](acceptance.md) | Completed solver-agnostic software and HPC gates |

Measurements are development records, not portable performance guarantees. Each entry must retain its platform, allocation, payload, build, and numerical comparison conditions.

Return to the [documentation index](../README.md).
