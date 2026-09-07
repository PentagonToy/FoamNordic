# OpenFOAM

## NACA4412 kEqn closure

Roihu interactive allocation `1095041` ran the 89,728-cell incompressible NACA4412 LES to physical time `1` with OpenFOAM v2512, 16 MPI ranks, `deltaT=0.0025` and 400 time steps. OpenFOAM via FoamNordic, abbreviated OF/FNO, used Longship to run the standard OpenFOAM `kEqn` model without an external closure. FNO-coupled used `kEqnFjord` with the same `Ck=0.094` and `Ce=1.048` equations evaluated as a resident float64 function over node-local SHM. Both paths received one warm-up followed by seven measured runs in alternating order on `rc5183` at commit `89cf240`. The unqualified name OpenFOAM is reserved for a direct OpenFOAM execution that does not pass through FoamNordic; the documentation index defines the abbreviations used here.

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `kEqn` | 34.891 ± 0.587 s | 30.350 ± 0.480 s | 30 ± 1 s |
| FNO-coupled `kEqnFjord` | 58.271 ± 0.793 s | 43.530 ± 0.830 s | 52 ± 1 s |

The measured end-to-end wall ratio was `1.670`; the OpenFOAM execution-time ratio was `1.434`. End-to-end wall includes Longship startup, readiness, solver execution and shutdown. These results establish the post-refactor baseline rather than a universal overhead claim, and expose the remaining lifecycle and per-exchange cost instead of folding it into model inference.

| Field | Relative L2 error | Peak-relative error | Result |
|---|---:|---:|---|
| `U` | 1.103e-3 | 1.978e-2 | PASS |
| `p` | 5.309e-3 | 4.968e-3 | PASS |
| `nut` | 6.966e-4 | 4.603e-3 | PASS |
| `k` | 6.785e-5 | 6.463e-5 | PASS |

Every measured pair produced the same final-field metrics. Acceptance required relative L2 error at or below `1e-2` and peak error relative to the reference peak at or below `5e-2`. A separate deterministic float64 algebra check over 4,096 rows matched `nut` and `kDissipationCoeff` exactly; `kProduction` differed only by a relative L2 of `1.172e-16`. This distinguishes algebraic agreement from the small differences accumulated when the external closure path is integrated for 400 nonlinear time steps.

## kEqn trajectory separation

The companion validation repeated OF/FNO and FNO-coupled runs from identical initial conditions at four independently restarted checkpoints. The values below are not successive samples from one run: each row represents a complete run from time zero to the stated step.

| Steps | Time | Field | Relative L2 error | Peak-relative error |
|---:|---:|---|---:|---:|
| 1 | 0.0025 | `U` | 8.315e-5 | 9.169e-4 |
| 1 | 0.0025 | `p` | 4.743e-4 | 2.564e-3 |
| 1 | 0.0025 | `nut` | 1.573e-4 | 2.178e-3 |
| 1 | 0.0025 | `k` | 2.880e-5 | 1.704e-4 |
| 10 | 0.025 | `U` | 2.176e-3 | 2.264e-2 |
| 10 | 0.025 | `p` | 1.462e-3 | 9.299e-3 |
| 10 | 0.025 | `nut` | 8.306e-4 | 6.289e-3 |
| 10 | 0.025 | `k` | 2.599e-4 | 1.622e-3 |
| 100 | 0.25 | `U` | 1.133e-3 | 1.821e-2 |
| 100 | 0.25 | `p` | 1.524e-2 | 2.535e-2 |
| 100 | 0.25 | `nut` | 1.138e-3 | 1.599e-2 |
| 100 | 0.25 | `k` | 5.479e-5 | 1.388e-4 |
| 400 | 1 | `U` | 1.103e-3 | 1.978e-2 |
| 400 | 1 | `p` | 5.309e-3 | 4.968e-3 |
| 400 | 1 | `nut` | 6.966e-4 | 4.603e-3 |
| 400 | 1 | `k` | 6.785e-5 | 6.463e-5 |

The separation is not monotonic. Pressure relative L2 reaches `1.524e-2` at the independently restarted 100-step checkpoint and returns to `5.309e-3` at 400 steps, while the other fields remain below `2.176e-3` throughout. This is evidence of a small time-dependent numerical trajectory difference, not evidence of steadily accumulating transport corruption. The runnable scripts are `tools/hpc/benchmarkNacaInteractive.sh` and `tools/hpc/validateNacaClosure.sh`; both preserve machine-readable JSON, and the validation also writes CSV and Markdown.
