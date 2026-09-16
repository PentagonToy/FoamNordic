# Python API reference

FoamNordic's installable Python control plane compiles immutable case, field-program, observation, placement, and scheduler declarations into a deterministic native plan. `Longship.launch()` prepares an isolated case and returns the cancellation-safe `Run`; `Run.stop()` produces one durable `Result`.

Python describes experiments before launch and inspects compact results during or after selected exchanges. Native code owns solver timing, field views, feature evaluation, scaling, inference, atomic publication, and failure handling. A Python callback is never inserted into a production solver loop.

| Reference | Public namespace | Contents |
| --- | --- | --- |
| [Run control](run-control.md) | `OpenFOAM`, `Operator`, `Closure`, `Transform`, `Observe`, `Longship`, `Slurm`, `Runtime` | Case declarations, field programmes, scheduling, launch, observation, cancellation and results |
| [Models and FNOM](model-artifact.md) | `Tensor`, `Export`, `Models` | ONNX, sklearn, Equinox, tables, preprocessing, inspection and validation |
| [Numerics](numerics.md) | `Field`, `Math`, `Context`, `Random` | Array algebra, symbolic OpenFOAM expressions, native graphs and random keys |
| [Postprocessing](postprocess.md) | `Postprocess` | Stored fields, statistics and case comparison |
| [Datasets](dataset.md) | `Dataset` | Lossless OpenFOAM field histories for machine learning and analysis |
| [Command-line interface](command-line.md) | `foamnordic` | Native installation, diagnostics, paths, FNOM tools and safe cleanup |

Public declarations are compiled before the solver starts. Explicit transport requests never silently fall back, observations cannot gate solver progress, and bulk field-program tensors do not pass through the notebook. Low-level contracts are documented under [architecture](../architecture/README.md); measured behaviour is kept under [benchmarks](../benchmarks/README.md). Wheel publication is a maintainer workflow documented in [others](../../others/release/publishing.md).

The interactive top-level namespace contains only the declarations used to define a workload or consume its results: `OpenFOAM`, `Closure`, `Transform`, `Observe`, `Longship`, `Operator`, `Field`, `Math`, `Context`, `Slurm`, `Export`, `Models`, `Tensor`, `Random`, `Runtime`, `Postprocess`, and `Dataset`. Lifecycle result types live under `Runtime`; model artefact types live under `Models`. Earlier lowercase modules and helper functions remain importable for compatibility but are not part of canonical interactive discovery, must not appear in new examples, and may be removed only in a declared major release.

## Reference conventions

Signatures show keyword-only arguments after `*`. A default of `None` means that FoamNordic either infers the value or disables the optional behaviour as described in the parameter table. Paths accept `str` and `os.PathLike`; relative paths are resolved by the owning declaration before launch. Public declarations are immutable unless a method explicitly documents an in-place state transition.

Parameter tables use `required` when no default exists. “Allowed values” lists closed sets enforced by the implementation; an omitted entry means that the Python type and stated validation define the accepted domain. Errors are raised during declaration or launch preparation whenever they can be detected before OpenFOAM starts.

The [command-line reference](command-line.md) documents installation, diagnostics, paths, FNOM tooling and marker-owned cleanup. Every command also provides independent `--help` output.
