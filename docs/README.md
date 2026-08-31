# FoamNordic documentation

The documentation is organized by the question a reader is trying to answer:

| Area | Contents |
| --- | --- |
| [Architecture](architecture/README.md) | Why FoamNordic is structured this way and how its processes are placed |
| [Python API](api/README.md) | How users declare cases, closures, schedules, observations, and postprocessing |
| [Native internals](internals/README.md) | How transport, inference, field exchange, and OpenFOAM integration work |
| [Benchmarks and validation](benchmarks/README.md) | What has been measured, on which platform, and which gates remain |

FoamNordic ships a native C++ data path and an installable Python control
plane. Architecture describes contracts, API documents describe public use,
internals describe implementation, and benchmark documents contain evidence.

Return to the [project README](../README.md).
