# FoamNordic documentation

**FNOM** is the portable FoamNordic Model artefact. In benchmark records, **OpenFOAM** is a direct run, **OF/FNO** is OpenFOAM launched by Longship without an external field programme, and **FNO-coupled** includes a closure or transform.

| Area | Use it for |
| --- | --- |
| [Python API](api/README.md) | Declare, execute, export, inspect, and analyse |
| [Architecture](architecture/README.md) | Runtime, transport, FNOM, OpenFOAM, and lifecycle contracts |
| [Developer guide](developer/README.md) | Source ownership, extension points, and verification |
| [Benchmarks](benchmarks/README.md) | Numerical parity and performance evidence |

The API documents supported use, architecture documents ownership and implementation, developer documents source changes and verification, and benchmarks preserve measured evidence. Return to the [project README](../README.md).

## Reading paths

| Goal | Start here | Continue with |
| --- | --- | --- |
| Define and launch a workload | [Run control](api/run-control.md) | [Numerics](api/numerics.md), [Models and FNOM](api/model-artifact.md) |
| Export and deploy a model | [Models and FNOM](api/model-artifact.md) | [FNOM architecture](architecture/fnom.md), [Runtime benchmarks](benchmarks/runtime.md) |
| Integrate an OpenFOAM solver | [OpenFOAM architecture](architecture/openfoam.md) | [Lifecycle](architecture/lifecycle.md), [Transport](architecture/transport.md) |
| Modify an exchange, closure, or backend | [Developer guide](developer/README.md) | [Field programmes](developer/field-programs.md), [Native runtime](developer/native-runtime.md) |
| Inspect stored results | [Postprocessing](api/postprocess.md) | [OpenFOAM validation](benchmarks/openfoam.md) |
| Install or diagnose the runtime | [Command-line interface](api/command-line.md) | [Native runtime](architecture/runtime.md) |
