# OpenFOAM case validation

This page records small, copied-case compatibility checks for the generic
FoamNordic field-program path. These checks answer whether an existing
OpenFOAM case can load `foamNordicExchange`, exchange a field, emit native
observations, and finish without changing the original case. They are software
integration checks, not physical validation of a turbulence model.

## macOS compatibility sweep

The 2026-08-22 sweep used OpenFOAM.com v2606 on Apple Silicon through the
[OpenFOAM app](https://github.com/gerlero/openfoam-app). The public case setup
was:

```python
case = fno.OpenFOAM.Case(
    name="caseName",
    case_dir=source,
    run_dir=output,
    of_cmd="openfoam",
    shell="zsh",
    application="pimpleFoam",
    ranks=1,
)
```

Each source case was copied to temporary storage. End times were shortened,
but the turbulence selection and original field dictionaries were retained.
An identity `U -> U` Transform ran at `time_step_start`, while native
observation summarized `U`, `p`, and `nut` where available. A second run of
the same copied case provided the uncoupled baseline.

| Regime | Case | Model or feature | Reduced gate | Result | Baseline comparison |
| --- | --- | --- | --- | --- | --- |
| Laminar | `incompressible/movingCone` | Dynamic mesh | 600 steps, serial | 600 exchanges and 600 observation records | final `U` and `p`: max absolute difference `0` |
| RAS | `incompressible/pitzDaily` | `kEpsilon` | 11 steps, serial | 11 exchanges; observed `U`, `p`, and `nut` | final `U`, `p`, and `nut`: max absolute difference `0` |
| LES | `incompressible/NACA4412` | `SpalartAllmarasDDES` | 1 step, serial | 1 exchange; observed `U`, `p`, and `nut` | final `U`, `p`, and `nut`: max absolute difference `0` |

The moving-mesh run completed in 8 seconds: 7 seconds in OpenFOAM and about
1 second in orchestration. The deliberately tiny RAS and LES gates completed
in 3 seconds each; their one-to-two-second orchestration share is dominated by
process startup and is not a throughput measurement.

The NACA4412 mesh retained its source-case `checkMesh` warnings for extreme
aspect ratio and non-orthogonality. They were not introduced by FoamNordic.

## What the sweep establishes

The same `foamNordicExchange` implementation works across laminar, RAS, and
LES cases. It does not replace `constant/turbulenceProperties`; only a declared
equation-level `Closure` renders a model-specific dictionary. This is why the
repository keeps `nutFjord` and `kEqnFjord` as LES model adapters without
creating empty laminar or RAS counterparts.

The identity comparisons also show that entering the exchange and observation
path does not perturb the tested fields. They do not establish accuracy for a
non-identity ML operator; that requires a model-specific reference and error
criteria.

## Known local limitation

The NACA4412 two-rank run failed during `MPI_Init` in the macOS OpenFOAM app.
The decomposition itself was complete: `numberOfSubdomains` was two and both
`processor0` and `processor1` were populated. The same `PML add procs failed`
error was then reproduced with a stock, uncoupled two-rank `pitzDaily` run, so
it is not a FoamNordic launch, model-adapter, or data-plane failure. The error
also matches [Open MPI issue #13129](https://github.com/open-mpi/ompi/issues/13129)
on newer macOS releases. Serial macOS remains the local compatibility gate.
Multi-rank OpenFOAM, including the native UCX closure path, is validated
separately on Linux HPC systems.

The `FPVFoam-v1912` corpus was intentionally excluded from this sweep. It is a
custom, release-specific combustion solver and will be treated as a dedicated
reaction-rate and table-coupling milestone rather than a generic field-program
smoke test.
