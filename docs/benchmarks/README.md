# Benchmarks and validation

These documents collect measured evidence and remaining acceptance gates.
They are separated from implementation descriptions so observed results are
not confused with architectural promises.

| Record | Contents |
|---|---|
| [Transport and performance](transport-and-performance.md) | macOS and Roihu transport measurements plus the OpenFOAM performance gate |
| [HPC transport](hpc-transport.md) | Implemented data planes, UCX environment, and portable site checks |
| [OpenFOAM cases](openfoam-cases.md) | Laminar, RAS, LES, combustion, and copied-case compatibility evidence |
| [Closure validation](closure-validation.md) | Mathematical parity and learned-closure accuracy and timing |
| [Compiled estimator](compiled-estimator.md) | Compiled C++ versus Joblib startup, memory, parity, and batch-size crossover |
| [Acceptance status](acceptance.md) | Completed software gates and remaining combustion acceptance work |

Measurements are development records, not portable performance guarantees.
Each entry must retain its platform, allocation, payload, build, and numerical
comparison conditions.

Return to the [documentation index](../README.md).
