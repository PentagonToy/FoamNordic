# FoamNordic documentation

The documentation is split by implementation boundary:

| Area | Contents |
| --- | --- |
| [Architecture](architecture/README.md) | Execution topologies, control/data planes, and model placement |
| [Native C++](native/README.md) | Transport, atomic exchange, inference, OpenFOAM integration, and placement |
| [Python API](python/README.md) | Current API status and the intended orchestration boundary |
| [Math API](python/math-api.md) | Backend-neutral NumPy/JAX and physical tensor operations |
| [Postprocess API](python/postprocess-api.md) | Durable OpenFOAM fields, statistics, and baseline/ML comparison |

FoamNordic ships its native C++ foundation and an installable Python control
plane. The run-control guide is executable API documentation; architecture
documents also record planned extensions where they are explicitly labelled.

Return to the [project README](../README.md).
