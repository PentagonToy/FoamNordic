# Native internals

These documents explain the implemented C++ hot path. They describe how data
moves and is evaluated; measured results belong in the benchmark section.

| Guide | Contents |
|---|---|
| [Data plane](data-plane.md) | Rune messages, Fjord channels, Harbor sessions, UDS, SHM, TCP, and UCX |
| [Closure engine](closure-engine.md) | Blocking exchanges, native bypass, resident execution, and failures |
| [Field pipeline](field-pipeline.md) | Zero-copy views, transforms, observations, and ownership |
| [Model artifacts](model-artifacts.md) | Manifests, scalers, ONNX, Equinox, Joblib, and preprocessing |
| [OpenFOAM adapter](openfoam-adapter.md) | Closure hooks, field operations, builds, and integration probes |
| [Combustion contract](combustion-contract.md) | Progress-variable, beta-FDF, thermodynamic, and adapter contracts |
| [Placement and lifecycle](placement-and-lifecycle.md) | Attached and central ClosureHost policies and transport selection |

The native path is the scientific hot path. Python orchestration must not add
per-cell callbacks or polling to it.

See also the [architecture](../architecture/README.md),
[Python API](../api/README.md), and
[validation evidence](../benchmarks/README.md).
