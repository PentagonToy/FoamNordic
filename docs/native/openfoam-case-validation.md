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

## Analytical LES parity

A separate 2026-08-22 gate compared stock OpenFOAM LES models against
mathematically equivalent float64 Equinox models executed through FoamNordic.
This is a closure-path parity test rather than a learned-model accuracy test:
there is no fitting error, and both runs use the same mesh, initial fields,
time scheme, pressure-velocity algorithm, filter width, and coefficients.

The Smagorinsky artifact implements the OpenFOAM.com relation

```text
D = symm(grad(U))
a = Ce/delta
b = (2/3) tr(D)
c = 2 Ck delta (dev(D) : D)
k = ((-b + sqrt(b^2 + 4ac))/(2a))^2
nut = Ck delta sqrt(k)
```

The k-equation artifact returns the three algebraic terms consumed by
`kEqnFjord`:

```text
nut               = Ck delta sqrt(k)
kProduction       = nut (grad(U) : dev(2 symm(grad(U))))
kDissipationCoeff = Ce sqrt(k)/delta
```

The transported k equation, discretization, relaxation, bounds, boundary
correction, and `fvOptions` remain in OpenFOAM. Both models use `Ck=0.094`,
`Ce=1.048`, and `cubeRootVol` filter width.

The test used the 16,384-cell lid-driven cavity for 100 steps from `t=0` to
`t=0.1` with `deltaT=0.001`, serial OpenFOAM.com v2606 on Apple Silicon, and
16-digit field output. Acceptance was `relative L2 <= 1e-9` and
`maxAbs <= 1e-12 + 1e-9*referenceMaxAbs`.

| Model | Field | Max absolute difference | Relative L2 | Result |
| --- | --- | ---: | ---: | --- |
| Smagorinsky | `U` | 6.737e-16 | 3.841e-16 | PASS |
| Smagorinsky | `p` | 1.679e-14 | 4.075e-13 | PASS |
| Smagorinsky | `nut` | 1.762e-19 | 5.985e-16 | PASS |
| kEqn | `U` | 5.634e-16 | 3.581e-16 | PASS |
| kEqn | `p` | 5.496e-15 | 6.156e-14 | PASS |
| kEqn | `nut` | 9.487e-20 | 2.384e-16 | PASS |
| kEqn | `k` | 3.123e-17 | 4.101e-16 | PASS |

The largest relative L2 difference was `4.075e-13` in Smagorinsky pressure;
the modeled `nut` fields agreed to relative L2 below `6e-16`. The complete
solver trajectories therefore agree to floating-point roundoff for this gate.

| Model | Stock total [s] | FoamNordic total [s] | Added wall [s] | Stock/FoamNordic OpenFOAM execution [s] |
| --- | ---: | ---: | ---: | ---: |
| Smagorinsky | 8.87 | 10.29 | 1.43 | 6.62 / 6.55 |
| kEqn | 8.87 | 10.58 | 1.71 | 6.76 / 6.77 |

The total includes process launch, readiness, resident JAX initialization and
JIT compilation, solver execution, and shutdown. OpenFOAM reported 7 seconds
of clock time for every run. This single short-case timing indicates that the
added wall time is predominantly fixed orchestration/model startup; it is not
a transport throughput benchmark.

The gate is reproducible with
[`tools/openfoam/analyticalLESParity.py`](../../tools/openfoam/analyticalLESParity.py).
It creates only isolated copies and writes a machine-readable
`analytical-les-parity.json` report.

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
