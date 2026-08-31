# Compiled estimator prototype

This development benchmark compares the `cpp-v1` compiled FNOM backend with
the existing uncompressed, directly mapped Joblib backend. It measures model
execution only; it is not an OpenFOAM wall-time result.

## Configuration

- Apple Silicon macOS, 8 model threads, 31 August 2026.
- 30,000 training rows and 10 float64 features.
- `VotingRegressor`: 64-tree `ExtraTreesRegressor` plus `Ridge`, weights
  `0.8 / 0.2`.
- Forest: maximum depth 16, minimum leaf size 2, 445,702 total tree nodes.
- Seven repetitions per warm prediction; table reports the median.
- Both artifacts use the same fitted estimator and no scaler.

The compiled predictor reproduces Joblib within `1.4e-15` maximum absolute
error. Its tree comparisons deliberately reproduce scikit-learn's float32
input narrowing before applying stored float64 thresholds.

## Artifact and startup

| Measurement | Joblib | Compiled `cpp-v1` |
|---|---:|---:|
| Export time [s] | 0.177 | 0.530 |
| FNOM size [MB] | 32.11 | 24.68 |
| First target compile [s] | n/a | 4.44 |
| Warm model load in an existing process [s] | 0.048 | 0.031 |
| Warm fresh-process load [s] | 1.17 | 0.031 |
| Warm fresh-process maximum RSS [MB] | 229 | 101 |

The first compiled fresh process took 2.51 s end to end and transiently reached
584 MB while Clang was active. Once cached, the same command took 0.11 s. The
first filesystem-cold Joblib process took 13.94 s and a repeated fresh process
took 1.42 s. Filesystem-cold numbers are especially host-dependent; the warm
fresh-process comparison is the useful deployment signal.

## Steady prediction

| Rows per exchange | Joblib [s] | Compiled [s] | Compiled speedup |
|---:|---:|---:|---:|
| 100 | 0.01354 | 0.00128 | 10.54x |
| 1,000 | 0.01367 | 0.00180 | 7.59x |
| 10,000 | 0.01366 | 0.01331 | 1.03x |
| 100,000 | 0.06666 | 0.05478 | 1.22x |

The compiled predictor uses native C++ row partitioning for small and medium
batches, then switches to tree partitioning at 32,768 rows. Tree-major traversal
improves cache locality and removes the former large-batch crossover: at 100,000
rows the native path is 2.25x faster than the earlier row-major C++ prototype
(0.12337 s) and 1.22x faster than Joblib. Creating native worker threads on each
call is included in every timing.

## Decision

Keep both backends. Compiled FNOM is the preferred path when the estimator graph
is supported and startup, resident memory, or repeated inference matters.
Joblib remains the compatibility backend for unsupported estimators. GPR and
custom estimator graphs are not claimed by `cpp-v1`.

## ExtraTrees and KNN voting

A second benchmark uses the same 30,000 by 10 training matrix, 64 ExtraTrees,
five distance-weighted neighbors, and equal `0.5 / 0.5` voting weights. Five
warm repetitions are reduced by their median. Exact KNN search is intentional;
both backends therefore retain linear dependence on the number of training
samples.

| Measurement | Joblib | Compiled `cpp-v1` |
|---|---:|---:|
| Export time [s] | 0.227 | 0.639 |
| FNOM size [MB] | 43.75 | 36.64 |
| Load or first target compile [s] | 0.052 | 2.343 |

| Rows per exchange | Joblib [s] | Compiled [s] | Compiled speedup |
|---:|---:|---:|---:|
| 100 | 0.02667 | 0.00679 | 3.93x |
| 1,000 | 0.03862 | 0.01234 | 3.13x |
| 10,000 | 0.20459 | 0.12332 | 1.66x |

The maximum difference from Joblib is `4.5e-16`. This result includes native
thread creation and confirms that a mixed forest/KNN voting graph can remain
inside one compiled FNOM without returning to Python between submodels.
