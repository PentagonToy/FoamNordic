# Native C++ documentation

These documents describe code implemented in the current repository.

| Guide | Contents |
| --- | --- |
| [Data plane](data-plane.md) | Rune messages, Fjord channels, Harbor sessions, UDS, SHM, TCP, and optional UCX |
| [HPC transport status](hpc-transport-status.md) | Implemented data planes, example site validation evidence, UCX environment, and a portable site checklist |
| [Closure engine](closure-engine.md) | Blocking exchanges, native bypass, resident execution, and failure semantics |
| [Model artifacts](model-artifacts.md) | Manifests, scalers, Equinox and Joblib boundaries, and ONNX Runtime |
| [Field pipeline](field-pipeline.md) | Zero-copy field views, native transforms, observations, and ownership |
| [OpenFOAM adapter](openfoam-adapter.md) | ClosureHook, field operations, build contract, and end-to-end probes |
| [OpenFOAM case validation](openfoam-case-validation.md) | Copied-case laminar, RAS, LES, analytical-closure, and combustion evidence |
| [Placement and lifecycle](placement-and-lifecycle.md) | Attached and central ClosureHost policies and transport selection |
| [Benchmarks](benchmarks.md) | Recorded transport measurements and the release performance gate |

The native path is the scientific hot path. Python orchestration must not add
per-cell callbacks or polling to it.

Return to the [documentation index](../README.md).
