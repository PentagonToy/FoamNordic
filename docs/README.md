# FoamNordic documentation

FoamNordic is abbreviated as **FNO** in tables and other space-constrained records. **FNOM** means **FoamNordic Model**, the portable saved-model artefact consumed by FoamNordic. The unqualified name OpenFOAM denotes a direct OpenFOAM execution, while **OF/FNO** denotes OpenFOAM launched through FoamNordic without an externally coupled field model and **FNO-coupled** denotes a run with a FoamNordic closure or transform connected to the solver.

| Area | Use it for |
| --- | --- |
| [Python API](api/README.md) | Cases, closures, models, Slurm, observations, and postprocessing |
| [Architecture](architecture/README.md) | Native runtime, transport, FNOM, OpenFOAM, and lifecycle contracts |
| [Benchmarks](benchmarks/README.md) | Numerical parity, performance measurements, and remaining validation gates |

The API documents describe supported use, architecture documents explain ownership and implementation, and benchmark documents preserve measured evidence. Return to the [project README](../README.md).

## Reading paths

| Goal | Start here | Continue with |
| --- | --- | --- |
| Define and launch a workload | [Run control API](api/run-control.md) | [Numerics](api/numerics.md), [FNOM artefacts](api/model-artifact.md) |
| Export and deploy a model | [FNOM artefact API](api/model-artifact.md) | [FNOM architecture](architecture/fnom.md), [runtime benchmarks](benchmarks/runtime.md) |
| Integrate another OpenFOAM solver | [OpenFOAM architecture](architecture/openfoam.md) | [Lifecycle](architecture/lifecycle.md), [transport](architecture/transport.md) |
| Inspect completed results | [Postprocess API](api/postprocess.md) | [OpenFOAM validation](benchmarks/openfoam.md) |
| Install or diagnose the runtime | [Command-line interface](api/command-line.md) | [Native runtime](architecture/runtime.md) |
