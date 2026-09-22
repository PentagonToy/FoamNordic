# Python API reference

FoamNordic's Python API declares a workload, compiles an immutable plan, launches an isolated OpenFOAM case, and consumes durable results. Python is a control plane; native code owns repeated solver execution, field access, scaling, inference, transport, and atomic publication.

## Public namespace

The top-level namespace is intentionally small. Grouped namespaces collect related operations; direct declarations define a workload.

| Group | Public names | Use it for |
| --- | --- | --- |
| Workload | `OpenFOAM`, `Field`, `Context`, `Math`, `Operator`, `Closure`, `Transform`, `Observe` | Declare case inputs, field programmes, and monitoring |
| Execution | `Longship`, `Slurm`, `Runtime` | Compile, place, launch, control, and inspect a run |
| Model artefacts | `Tensor`, `Export`, `Models` | Export, inspect, validate, and deploy FNOM artefacts |
| Results | `Postprocess` | Read fields and compare runs |
| Reproducibility | `Random` | Declare deterministic stochastic programmes |

Lowercase modules and older helper names remain importable only for compatibility. New code should use the names above.

## Reference map

| Task | Reference | Main API |
| --- | --- | --- |
| Declare and launch an OpenFOAM workload | [Run control](run-control.md) | `OpenFOAM`, `Operator`, `Closure`, `Transform`, `Observe`, `Longship`, `Slurm`, `Runtime` |
| Build symbolic field programmes | [Numerics](numerics.md) | `Field`, `Context`, `Math`, `Random` |
| Export or inspect a model or table | [Models and FNOM](model-artifact.md) | `Tensor`, `Export`, `Models` |
| Read, summarise, or compare stored fields | [Postprocessing](postprocess.md) | `Postprocess` |
| Install, inspect, or maintain a runtime | [Command-line interface](command-line.md) | `foamnordic` |

## Reference conventions

Signatures show keyword-only options after `*`. Parameters follow the repository convention: subject, selection or configuration, destination, execution or reporting, then safety control. `None` either disables an optional behaviour or requests documented inference. Paths accept `str` and `os.PathLike`.

Declarations are immutable unless a method explicitly documents an in-place transition. Validation failures are raised during declaration or launch preparation whenever they can be detected before OpenFOAM starts. Architecture contracts are in [Architecture](../architecture/README.md), measured evidence is in [Benchmarks](../benchmarks/README.md), and publication maintenance is in [Release publishing](../../others/release/publishing.md).
