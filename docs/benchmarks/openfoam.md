# OpenFOAM

This record compares solver-native FoamNordic functions with their standard OpenFOAM reference models. It isolates framework overhead and numerical parity; it does not assess the predictive quality of a learned closure.

## Metrics

For a reference field $\mathbf{r}$ and FoamNordic field $\mathbf{f}$, the relative $L_2$ error is

$$
E_{L_2}=\frac{\lVert\mathbf{f}-\mathbf{r}\rVert_2}{\lVert\mathbf{r}\rVert_2},
$$

and the reported peak-relative error is

$$
E_{\infty}=\frac{\lVert\mathbf{f}-\mathbf{r}\rVert_\infty}{\max(\lVert\mathbf{r}\rVert_\infty,s_{\min})}.
$$

The scale floor $s_{\min}$ is the smallest positive float for the pitzDaily comparisons and $10^{-12}$ for NACA4412. Performance tables report the median and maximum absolute deviation defined in the [benchmark reporting contract](README.md). Ratios use FoamNordic divided by the corresponding OF/FNO value; a ratio of $1$ denotes equal elapsed time.

## Validation matrix

| Case | Reference | FoamNordic path | Cells | MPI ranks | Steps |
| --- | --- | --- | ---: | ---: | ---: |
| pitzDaily | Smagorinsky | FNO-native `nutFjord` | Case-defined | 16 | 1, 10, 100, 400 |
| pitzDaily | WALE | FNO-native `nutFjord` | Case-defined | 16 | 1, 10, 100, 400 |
| NACA4412 | kEqn | FNO-native `kEqnFjord` | 89,728 | 16 | 1, 10, 100, 400 |

## pitzDaily `nutFjord`

Roihu compared OF/FNO with FNO-native `nutFjord` on the incompressible pitzDaily case using OpenFOAM v2512, 16 MPI ranks and deterministic `simple (16 1 1)` decomposition. Smagorinsky and WALE were each restarted from identical initial conditions and run independently for 1, 10, 100 and 400 steps with `deltaT=1e-5`.

### Smagorinsky

The standard OpenFOAM Smagorinsky model used `Ck=0.0265463553` and `Ce=1.048`; FNO-native evaluated the same float64 algebra through `nutFjord` without a ModelHost or field exchange.

#### Performance

Each path received one warm-up followed by seven measured 400-step runs in alternating order.

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `Smagorinsky` | 7.309 ± 0.147 s | 4.140 ± 0.100 s | 4 ± 0 s |
| FNO-native `nutFjord` | 7.847 ± 0.227 s | 4.160 ± 0.250 s | 4 ± 0 s |

The end-to-end wall ratio was `1.074`; the OpenFOAM execution-time ratio was `1.005`. All seven measured pairs produced identical final fields.

#### Relative L2 error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 0 | 0 | 0 | 0 |
| `p` | 0 | 0 | 0 | 0 |
| `nut` | 0 | 0 | 0 | 0 |

#### Peak-relative error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 0 | 0 | 0 | 0 |
| `p` | 0 | 0 | 0 | 0 |
| `nut` | 0 | 0 | 0 | 0 |

### WALE

The standard OpenFOAM WALE model used `Cw=0.325`; FNO-coupled evaluated the same float64 velocity-gradient invariant expression through `nutFjord`.

#### Performance

Each path received one warm-up followed by seven measured 400-step runs in alternating order.

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `WALE` | 7.229 ± 0.185 s | 4.060 ± 0.150 s | 4 ± 0 s |
| FNO-native `nutFjord` | 7.934 ± 0.263 s | 4.200 ± 0.220 s | 4 ± 1 s |

The end-to-end wall ratio was `1.097`; the OpenFOAM execution-time ratio was `1.034`. All seven measured pairs produced identical final-field metrics.

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

The columns correspond to physical times `1e-5`, `1e-4`, `1e-3` and `4e-3`. All values were finite and the direct float64 functions remained nonnegative. The runnable scripts are `tools/hpc/benchmarkSmagorinskyInteractive.sh`, `tools/hpc/benchmarkWaleInteractive.sh` and `tools/hpc/validateNutClosure.sh`.

## NACA4412 `kEqnFjord`

### k-equation configuration

Both NACA4412 paths ran on Roihu with 89,728 cells, OpenFOAM v2512, 16 MPI ranks, `deltaT=0.0025` and 400 steps to physical time `1`.

OF/FNO launched standard OpenFOAM `kEqn` through Longship. FNO-native used identical `Ck=0.094` and `Ce=1.048` algebra in `Operator.function`, with the implicit equation in `Field.transport`. The lowered `kEqnFjord` graph ran at the correction boundary without ModelHost or field exchange. Each path used one warm-up and seven measured runs, alternating execution order.

#### Performance

| Path | End-to-end wall | OpenFOAM execution | OpenFOAM clock |
|---|---:|---:|---:|
| OF/FNO `kEqn` | 37.002 ± 1.868 s | 31.670 ± 1.850 s | 32 ± 2 s |
| FNO-native `kEqnFjord` | 39.919 ± 2.915 s | 33.920 ± 3.010 s | 34 ± 3 s |

The end-to-end wall ratio was `1.079`; the OpenFOAM execution-time ratio was `1.071`. End-to-end wall includes Longship preparation, solver execution and shutdown.

#### Relative L2 error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 6.673e-6 | 1.522e-3 | 1.470e-3 | 1.726e-3 |
| `p` | 3.745e-4 | 2.692e-3 | 3.693e-3 | 7.803e-3 |
| `nut` | 2.831e-6 | 2.919e-4 | 1.090e-3 | 9.401e-4 |
| `k` | 2.123e-6 | 4.664e-5 | 4.526e-5 | 2.307e-4 |

#### Peak-relative error

| Field | 1 step | 10 steps | 100 steps | 400 steps |
|---|---:|---:|---:|---:|
| `U` | 5.632e-5 | 1.699e-2 | 3.376e-2 | 2.356e-2 |
| `p` | 1.426e-3 | 2.047e-2 | 2.613e-2 | 8.056e-3 |
| `nut` | 3.377e-5 | 1.372e-3 | 2.177e-2 | 7.042e-3 |
| `k` | 1.512e-5 | 1.275e-4 | 1.282e-4 | 1.716e-4 |

The columns correspond to physical times `0.0025`, `0.025`, `0.25` and `1`. Every measured pair produced the same final-field metrics and passed the declared relative L2 tolerance of `1e-2` and peak-relative tolerance of `5e-2`. The runnable benchmark is `tools/hpc/benchmarkNacaInteractive.sh`.
