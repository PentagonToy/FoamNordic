# OpenFOAM

## pitzDaily nutFjord closure

Roihu interactive allocation `1095041` compared OF/FNO with FNO-coupled `nutFjord` on the incompressible pitzDaily case using OpenFOAM v2512, 16 MPI ranks and deterministic `simple (16 1 1)` decomposition. Smagorinsky and WALE were each restarted from identical initial conditions and run independently for 1, 10, 100 and 400 steps with `deltaT=1e-5`. These single-run timings are diagnostic and are not treated as a performance benchmark.

### Smagorinsky

The standard OpenFOAM Smagorinsky model used `Ck=0.0265463553` and `Ce=1.048`; FNO-coupled evaluated the same float64 algebra through `nutFjord`.

#### Performance

Each path received one warm-up followed by seven measured 400-step runs in alternating order.

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `Smagorinsky` | 7.568 ± 0.282 s | 4.270 ± 0.150 s | 4 ± 1 s |
| FNO-coupled `nutFjord` | 12.675 ± 0.236 s | 7.600 ± 0.300 s | 8 ± 1 s |

The end-to-end wall ratio was `1.675`; the OpenFOAM execution-time ratio was `1.780`.

#### Relative L2 error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 0 | 7.910e-8 | 2.546e-7 | 1.083e-7 |
| `p` | 0 | 1.754e-8 | 1.798e-8 | 4.306e-8 |
| `nut` | 1.041e-16 | 1.861e-7 | 3.245e-7 | 1.835e-7 |

#### Peak-relative error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 0 | 1.312e-6 | 2.313e-6 | 5.652e-7 |
| `p` | 0 | 7.394e-7 | 3.188e-7 | 2.974e-7 |
| `nut` | 4.320e-16 | 3.069e-7 | 7.808e-7 | 1.458e-7 |

### WALE

The standard OpenFOAM WALE model used `Cw=0.325`; FNO-coupled evaluated the same float64 velocity-gradient invariant expression through `nutFjord`.

#### Relative L2 error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 1.060e-13 | 6.722e-12 | 6.140e-12 | 4.747e-12 |
| `p` | 6.907e-14 | 2.514e-12 | 3.573e-13 | 1.053e-12 |
| `nut` | 2.198e-6 | 2.988e-6 | 1.498e-6 | 1.620e-6 |

#### Peak-relative error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 1.525e-13 | 4.388e-11 | 5.181e-11 | 4.323e-11 |
| `p` | 5.267e-14 | 6.376e-12 | 2.318e-12 | 3.211e-12 |
| `nut` | 1.597e-6 | 2.380e-6 | 3.246e-6 | 4.146e-6 |

The columns correspond to physical times `1e-5`, `1e-4`, `1e-3` and `4e-3`. All values were finite and the direct float64 functions remained nonnegative. The runnable scripts are `tools/hpc/benchmarkSmagorinskyInteractive.sh` and `tools/hpc/validateNutClosure.sh`.

## NACA4412 kEqn closure

Roihu interactive allocation `1095041` ran the 89,728-cell incompressible NACA4412 LES to physical time `1` with OpenFOAM v2512, 16 MPI ranks, `deltaT=0.0025` and 400 time steps. OpenFOAM via FoamNordic, abbreviated OF/FNO, used Longship to run the standard OpenFOAM `kEqn` model without an external closure. FNO-coupled used `kEqnFjord` with the same `Ck=0.094` and `Ce=1.048` equations evaluated as a resident float64 function over node-local SHM. Both paths received one warm-up followed by seven measured runs in alternating order on `rc5183` at commit `89cf240`. The unqualified name OpenFOAM is reserved for a direct OpenFOAM execution that does not pass through FoamNordic; the documentation index defines the abbreviations used here.

### Performance

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `kEqn` | 34.891 ± 0.587 s | 30.350 ± 0.480 s | 30 ± 1 s |
| FNO-coupled `kEqnFjord` | 58.271 ± 0.793 s | 43.530 ± 0.830 s | 52 ± 1 s |

The end-to-end wall ratio was `1.670`; the OpenFOAM execution-time ratio was `1.434`. End-to-end wall includes Longship startup, readiness, solver execution and shutdown.

### Final-field agreement

| Field | Relative L2 error | Peak-relative error |
|---|---:|---:|
| `U` | 1.103e-3 | 1.978e-2 |
| `p` | 5.309e-3 | 4.968e-3 |
| `nut` | 6.966e-4 | 4.603e-3 |
| `k` | 6.785e-5 | 6.463e-5 |

Every measured pair produced the same final-field metrics. A deterministic float64 algebra check over 4,096 rows matched `nut` and `kDissipationCoeff` exactly; `kProduction` had relative L2 error `1.172e-16`.

### Trajectory separation

The companion validation repeated OF/FNO and FNO-coupled runs from identical initial conditions at four independently restarted checkpoints. The values below are not successive samples from one run: each row represents a complete run from time zero to the stated step.

#### Relative L2 error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 8.315e-5 | 2.176e-3 | 1.133e-3 | 1.103e-3 |
| `p` | 4.743e-4 | 1.462e-3 | 1.524e-2 | 5.309e-3 |
| `nut` | 1.573e-4 | 8.306e-4 | 1.138e-3 | 6.966e-4 |
| `k` | 2.880e-5 | 2.599e-4 | 5.479e-5 | 6.785e-5 |

#### Peak-relative error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 9.169e-4 | 2.264e-2 | 1.821e-2 | 1.978e-2 |
| `p` | 2.564e-3 | 9.299e-3 | 2.535e-2 | 4.968e-3 |
| `nut` | 2.178e-3 | 6.289e-3 | 1.599e-2 | 4.603e-3 |
| `k` | 1.704e-4 | 1.622e-3 | 1.388e-4 | 6.463e-5 |

The columns correspond to physical times `0.0025`, `0.025`, `0.25` and `1`, respectively.

The separation is not monotonic: pressure relative L2 is `1.524e-2` at 100 steps and `5.309e-3` at 400 steps. The runnable scripts are `tools/hpc/benchmarkNacaInteractive.sh` and `tools/hpc/validateNacaClosure.sh`.
