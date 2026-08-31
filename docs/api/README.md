# Python API documentation

FoamNordic's installable Python control plane compiles immutable case, closure,
observation, placement, and scheduler declarations into a deterministic native
plan. `Longship.launch()` prepares an isolated case and returns the
cancellation-safe `Run`; `Run.stop()` produces one durable `Result`.

Python describes experiments before launch and inspects compact results during
or after selected exchanges. Native code owns solver timing, field views,
feature evaluation, scaling, inference, atomic publication, and failure
handling. A Python callback is never inserted into a production closure loop.

| Guide | Contents |
| --- | --- |
| [Run control](run-control-api.md) | cases, closures, transforms, observations, Slurm, and lifecycle |
| [Models and FNOM](model-artifact-api.md) | ONNX, sklearn, Equinox, preprocessing, inspection, and validation |
| [Numerics](numerics.md) | `fno.Math`, OpenFOAM expressions, and reproducible random keys |
| [Combustion](combustion-api.md) | progress-variable and beta-FDF declarations |
| [Postprocessing](postprocess-api.md) | stored fields, statistics, and case comparisons |

Public declarations are compiled before the solver starts. Explicit transport
requests never silently fall back, observations cannot gate solver progress,
and bulk closure fields do not pass through the notebook. Low-level contracts
are documented under [architecture](../architecture/README.md); measured
behavior is kept under [benchmarks](../benchmarks/README.md). Wheel publication
is a maintainer workflow documented in [others](../../others/publishing.md).
