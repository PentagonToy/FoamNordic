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

## Equation-level closure evidence

Mathematical Smagorinsky and k-equation parity, together with the Linux
pitzDaily mathematical-versus-Joblib experiment, now live in
[Mathematical and learned closure validation](closure-validation.md). Keeping
that evidence separate makes the distinction between generic case
compatibility, exact equation parity, and learned-model quality explicit.

## Laminar combustion field parity

A 2026-08-22 combustion gate used the OpenFOAM.com v2606
`counterFlowFlame2DLTS_GRI_TDAC` case on Apple Silicon. This is a 4,000-cell
laminar `reactingFoam` case with 53 species, 325 reactions, the TDAC chemistry
method, DAC reduction, and ISAT tabulation. The copied case ran for three LTS
steps. No source case was modified.

The first FoamNordic run exercised the pure OpenFOAM path. A second run sent
`T` and `CH4` through one identity `Transform` at `time_step_start`, returned
both fields atomically, and observed their minimum, maximum, and mean at every
exchange. This tests a compressible reacting solver, multicomponent field
discovery, two-field resident execution, SHM transport, native observation,
and postprocessing without claiming an equation-level reaction-rate closure.

Acceptance was `relative L2 <= 1e-9` and
`maxAbs <= 1e-12 + 1e-9*referenceMaxAbs`.

| Field | Max absolute difference | Relative L2 | Result |
| --- | ---: | ---: | --- |
| `U` | 5.965e-11 | 8.514e-11 | PASS |
| `p` | 1.892e-10 | 4.094e-16 | PASS |
| `T` | 1.137e-12 | 2.927e-17 | PASS |
| `CH4` | 3.469e-17 | 1.057e-17 | PASS |
| `O2` | 9.926e-24 | 1.193e-23 | PASS |
| `CO2` | 0 | 0 | PASS |
| `H2O` | 0 | 0 | PASS |
| `Qdot` | 0 | 0 | PASS |

The generated chemistry heat-release field `Qdot` agrees exactly, while the
largest relative L2 difference is `8.514e-11` in velocity. This establishes
that the current generic field-program path can operate inside a real
finite-rate reacting case without materially perturbing its three-step
trajectory.

| Measurement | Stock | FoamNordic identity |
| --- | ---: | ---: |
| End-to-end wall time | 2.03 s | 5.06 s |
| OpenFOAM `ExecutionTime` | 0.11 s | 0.11 s |

The first exchange waited 15.6 ms for the resident; subsequent closure waits
were 0.17--0.21 ms and Python function evaluation took 11.7--13.7 us. The
approximately three-second wall difference is fixed worker and lifecycle
startup on a solver that itself runs for only 0.11 seconds. It is not a
representative long-run overhead ratio.

The gate is reproducible with
[`tools/openfoam/combustionFieldParity.py`](../../tools/openfoam/combustionFieldParity.py).
It prepares an isolated mesh, writes all runs beneath the selected output
directory, and emits `combustion-field-parity.json`.

### Equation-level boundary

The generic parity gate was followed by a native `reactionRateFjord`
`ThermoCombustion` adapter. It now evaluates a learned scalar source at the
combustion model's correction site and validates its solver-owned field and
dimensions. A generic `Transform` remains appropriate for deliberate field
modification, while this adapter establishes the source-evaluation boundary.

The boundary was then extended with `progressVariableFjord`: two resident FNOM
programs are lowered from the public combustion declaration, `Y_*` output
families are expanded before launch, and the native coordinator executes
reaction rate, manifold, and one thermodynamic correction in that order. The
model compiles and registers for both `psiReactionThermo` and
`rhoReactionThermo` with OpenFOAM.com v2606.

The equation boundary now also exposes the configured source through
`combustion->R(progress)`. A compiled v2606 probe checked the resulting matrix
on all 16,384 cells of the cavity mesh: dimensions matched the configured
volumetric source times cell volume, the positive right-hand-side convention
was preserved, and the maximum difference from `V*omega_c` was zero. The
cavity is only a mesh-level source gate, not a combustion validation case.

The local complete-combustion integration now includes a solver-integrated
beta-FDF artifact fixture, an atomic field trajectory, a stock reacting-solver
adapter gate, and exact volumetric/specific source probes. The remaining
software acceptance is the Linux HPC/OpenFOAM v2512 matrix documented in
`acceptance.md`. Boundary conservation, realizability, failure
policy, and a physically representative trained artifact remain scientific
acceptance work.

The shared Fjord, resident, artifact, scaling, observation, lifecycle,
reaction-rate, source-matrix, and manifold-dispatch layers are now present.
Solver-specific transport equations and their physical acceptance trajectory
remain.

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
