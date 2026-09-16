# FoamNordic documentation

FoamNordic is abbreviated as **FNO** in tables and other space-constrained records. **FNOM** means **FoamNordic Model**, the portable saved-model artefact consumed by FoamNordic. The unqualified name OpenFOAM denotes a direct OpenFOAM execution, while **OF/FNO** denotes OpenFOAM launched through FoamNordic without an externally coupled field model and **FNO-coupled** denotes a run with a FoamNordic closure or transform connected to the solver.

| Area | Use it for |
| --- | --- |
| [Python API](api/README.md) | Cases, closures, models, Slurm, observations, postprocessing, and datasets |
| [Architecture](architecture/README.md) | Native runtime, transport, FNOM, OpenFOAM, and lifecycle contracts |
| [Developer guide](developer/README.md) | Source ownership, extension points, change routes, and verification |
| [Benchmarks](benchmarks/README.md) | Numerical parity, performance measurements, and remaining validation gates |

The API documents describe supported use, architecture documents explain ownership and implementation, developer documents route source changes and verification, and benchmark documents preserve measured evidence. Return to the [project README](../README.md).

## Reading paths

| Goal | Start here | Continue with |
| --- | --- | --- |
| Define and launch a workload | [Run control API](api/run-control.md) | [Numerics](api/numerics.md), [FNOM artefacts](api/model-artifact.md) |
| Export and deploy a model | [FNOM artefact API](api/model-artifact.md) | [FNOM architecture](architecture/fnom.md), [runtime benchmarks](benchmarks/runtime.md) |
| Integrate another OpenFOAM solver | [OpenFOAM architecture](architecture/openfoam.md) | [Lifecycle](architecture/lifecycle.md), [transport](architecture/transport.md) |
| Modify an exchange, closure, or backend | [Developer guide](developer/README.md) | [Field programmes](developer/field-programs.md), [native runtime](developer/native-runtime.md) |
| Inspect completed results | [Postprocess API](api/postprocess.md) | [OpenFOAM validation](benchmarks/openfoam.md) |
| Build an ML dataset from OpenFOAM fields | [Dataset API](api/dataset.md) | [FNOM artefacts](api/model-artifact.md), [Numerics](api/numerics.md) |
| Install or diagnose the runtime | [Command-line interface](api/command-line.md) | [Native runtime](architecture/runtime.md) |
