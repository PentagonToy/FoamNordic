# Mathematical and learned closure validation

This page records equation-level validation of mathematical and learned LES
closures. General copied-case compatibility evidence remains in
[OpenFOAM case validation](openfoam-case-validation.md). These gates distinguish
software-path parity from the scientific accuracy and runtime suitability of a
fitted model.

## Analytical LES parity

These gates compare stock OpenFOAM LES models with the same equations executed
through FoamNordic. They test the closure path rather than learned-model
accuracy: there is no fitting error, and each pair preserves its mesh, initial
fields, numerical schemes, pressure-velocity algorithm, filter width, and
coefficients.

### Adapter contracts and equations

`nutFjord` is not a turbulence equation of its own. It is the general
OpenFOAM adapter for algebraic LES closures with the contract
`grad(U)[9], delta[1] -> nut[1]`. Mathematical Smagorinsky and WALE both use
this adapter by supplying a different `Operator.function()` while OpenFOAM
continues to own boundary correction and momentum-equation coupling.

The Smagorinsky function implements the OpenFOAM.com relation

```text
D = symm(grad(U))
a = Ce/delta
b = (2/3) tr(D)
c = 2 Ck delta (dev(D) : D)
k = ((-b + sqrt(b^2 + 4ac))/(2a))^2
nut = Ck delta sqrt(k)
```

The WALE function uses

```text
G   = grad(U)
S   = symm(G)
Sd  = dev(symm(G G))
sd2 = Sd : Sd
s2  = S : S
nut = (Cw delta)^2 sd2^(3/2) / (s2^(5/2) + sd2^(5/4))
```

When the denominator is zero, the WALE function returns its analytical limit,
`nut=0`, without introducing an arbitrary absolute epsilon.

`kEqnFjord` has a different contract because OpenFOAM transports `k`. Its
function returns the three algebraic terms consumed by that equation:

```text
nut               = Ck delta sqrt(k)
kProduction       = nut (grad(U) : dev(2 symm(grad(U))))
kDissipationCoeff = Ce sqrt(k)/delta
```

The transported k equation, discretization, relaxation, bounds, boundary
correction, and `fvOptions` remain in OpenFOAM. The Smagorinsky and k-equation
gates use `Ck=0.094`, `Ce=1.048`, and `cubeRootVol` filter width.

### Apple lid-driven-cavity parity

The 2026-08-22 gate compared stock Smagorinsky and k-equation models against
mathematically equivalent float64 Equinox artifacts executed through
FoamNordic.

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

### Linux pitzDaily WALE parity

A 2026-08-24 serial OpenFOAM.com v2512 comparison exercised WALE through the
same general `nutFjord` contract used by the mathematical Smagorinsky closure.
Both paths used the 12,225-cell `pitzDailyWALE` mesh, `pimpleFoam`,
`cubeRootVol`, `Cw=0.325`, `deltaT=1e-5`, and 2,000 steps from `t=0` to
`t=0.02`. The stock path selected OpenFOAM `WALE`; the FoamNordic path used a
resident `Operator.function()` with the contract `grad(U)[9], delta[1] ->
nut[1]`.

The zero-denominator policy defined above keeps the result independent of
floating-point precision, field scale, and MPI decomposition.

The final fields at `t=0.02` were compared over every cell:

| Field | MAE | RMSE | Max absolute difference | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| `U` | 0 | 0 | 0 | 0 |
| `p` | 0 | 0 | 0 | 0 |
| `nut` | 2.456e-16 | 2.652e-14 | 2.932e-12 | 2.398e-9 |

The velocity and pressure files therefore agree exactly at their retained
precision. The eddy-viscosity difference remains below `2.4e-9` in relative
L2 and is consistent with evaluation-order roundoff across the native WALE
and resident function implementations.

| Path | Started (Europe/Helsinki) | Host | OpenFOAM ExecutionTime [s] | OpenFOAM ClockTime [s] | FoamNordic total [s] | Orchestration [s] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Stock WALE | 18:14:28 | `roihu-cpu-login1` | 47.17 | 49 | n/a | n/a |
| Mathematical WALE through `nutFjord` | 18:28:26 | `rc5283` | 52.21 | 56 | 57 | 1 |

The hosts differ, so the seven-second solver wall-time difference is only
indicative. The FoamNordic timing does verify the corrected normal-shutdown
lifecycle: after OpenFOAM completed, Longship signalled the resident
ClosureHost immediately and finalized in one second instead of consuming the
30-second termination grace period before sending the signal.

## Linux pitzDaily learned closure

A 2026-08-23 Linux/OpenFOAM.com v2512 experiment used the analytical
Smagorinsky result as a reference for learned closures on the 12,225-cell
`pitzDaily` case. All five serial trajectories used
`pimpleFoam`, `cubeRootVol`, `Ck=0.0265463553`, `Ce=1.048`,
`deltaT=1e-5`, and 2,000 steps from `t=0` to `t=0.02`. `blockMesh`
produced the same strict mesh for every comparison and `checkMesh` reported
`Mesh OK`.

The stock trajectory used OpenFOAM `Smagorinsky` directly. The mathematical
FoamNordic trajectory used the same relation in a resident
`Operator.function()` bound through `nutFjord`. Two learned trajectories used
scikit-learn estimators exported through the Joblib FNOM backend: a composite
`VotingRegressor` and an `ExtraTreesRegressor`. A third used a two-hidden-layer
Equinox MLP. All FoamNordic runs used rank-local SHM and loaded their resident
payload once.

### Learned-model contract

Vanilla pitzDaily fields at physical times `0.0025` through `0.0175`
provided training trajectories, while `t=0.02` remained held out. Each sample
contained the nine components of `grad(U)` and `delta=cbrt(V)`; the target
was the analytical Smagorinsky `nut`. Thirty thousand samples were retained
by stratifying over target deciles.

The ten-feature `StandardScaler` was fitted before estimator training.
Features with variance below `1e-20`, which arise naturally from inactive
components of this two-dimensional case, were deliberately left unscaled by
setting their exported mean to zero and scale to one. Both estimators were
fitted on the same scaled features and exported with that same scaler. The
Extra Trees model contained 100 estimators. The Voting model combined those
trees with five uniform nearest neighbors using weights of 0.8 and 0.2.
Both used the `grad(U)[9], delta[1] -> nut[1]` contract and selected the
`sklearnex` resident runtime. Their Joblib payloads were approximately 180
MiB and 186 MiB, respectively.

The Equinox MLP used the same input samples and held-out physical time. It
standardized both inputs and the scalar target, used two 64-unit ReLU hidden
layers, and optimized mean-squared error with Adam. The exported 32.6 KiB
payload included the fitted input and output scalers and used the same
`grad(U)[9], delta[1] -> nut[1]` contract. Its resident inference path applied
the inverse target transform but did not include the nonnegative clipping used
by the notebook's offline metric calculation.

The Voting model is retained deliberately as a bottleneck gate rather than a
recommended production closure. It represents a more complex or customized
regressor that combines heterogeneous estimators behind one Joblib contract.
Comparing it with the isolated Extra Trees component measures how estimator
composition affects end-to-end solver latency and trajectory quality while
the FNOM contract, scaler, resident process, SHM transport, mesh, and solver
remain fixed.

### Final-field comparison at t=0.02

The comparison uses the stock trajectory as the reference and verifies the
mesh byte-for-byte before comparing all 12,225 cell values.

| Closure | Field | MAE | RMSE | Max absolute | Relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Mathematical `nutFjord` | `U` | 1.249e-6 | 5.966e-6 | 1.044e-4 | 8.647e-7 |
| Mathematical `nutFjord` | `p` | 2.519e-6 | 1.578e-5 | 1.000e-4 | 2.300e-7 |
| Mathematical `nutFjord` | `nut` | 1.416e-12 | 6.238e-12 | 1.000e-10 | 9.999e-7 |
| Learned Joblib `VotingRegressor` | `U` | 4.498e-2 | 1.909e-1 | 6.753 | 4.792e-2 |
| Learned Joblib `VotingRegressor` | `p` | 3.930e-1 | 1.751 | 28.458 | 2.552e-2 |
| Learned Joblib `VotingRegressor` | `nut` | 3.754e-7 | 1.432e-6 | 4.469e-5 | 2.296e-1 |
| Learned Joblib `ExtraTrees` | `U` | 4.488e-2 | 1.902e-1 | 6.659 | 4.774e-2 |
| Learned Joblib `ExtraTrees` | `p` | 3.841e-1 | 1.712 | 27.882 | 2.495e-2 |
| Learned Joblib `ExtraTrees` | `nut` | 3.524e-7 | 1.399e-6 | 4.411e-5 | 2.243e-1 |
| Learned Equinox MLP | `U` | 9.966e-3 | 3.293e-2 | 6.202e-1 | 8.265e-3 |
| Learned Equinox MLP | `p` | 7.157e-2 | 1.778e-1 | 2.647 | 2.592e-3 |
| Learned Equinox MLP | `nut` | 3.406e-7 | 1.174e-6 | 3.368e-5 | 1.881e-1 |

The mathematical adapter reproduces the stock trajectory to approximately
the precision retained by these OpenFOAM field files. This independently
checks the v2512 `nutFjord` equation boundary, tensor restoration,
`grad(U)`, LES `delta`, SHM exchange, and resident function execution.

The learned models are not parity results. Extra Trees slightly improved all
three whole-field relative L2 errors over the Voting model. Its correlations
were `0.998596` for `U`, `0.996927` for `p`, and `0.970657` for `nut`, compared
with `0.998585`, `0.996787`, and `0.969174` for Voting. Extra Trees component
relative L2 errors were `3.994e-2` for `Ux`, `1.203e-1` for `Uy`, and
`2.990e-3` for the numerically negligible `Uz`. Pressure had a mean shift of
`5.792e-2`; after removing each field's mean, its relative L2 difference was
`7.834e-2`.

Both learned `nut` fields remained nonnegative in every cell. The Extra Trees
mean was `2.382e-6`, close to the stock `2.312e-6`, while its maximum was
`9.113e-5` versus the stock `1.352e-4`. Its active-region 99th-percentile
relative error was `1.443`, compared with `1.558` for Voting. Extra Trees is
therefore both the faster and slightly more accurate learned trajectory in
this gate, although its `2.243e-1` `nut` relative L2 error does not qualify it
as mathematical parity or, by itself, as a production LES closure.

The Equinox MLP improved the whole-field errors further. Its correlations
were `0.999958` for `U`, `0.999968` for `p`, and `0.979911` for `nut`.
Component relative L2 errors were `6.775e-3` for `Ux`, `2.164e-2` for `Uy`,
and `4.178e-4` for the numerically negligible `Uz`. Pressure had a mean shift
of `1.345e-2`; after removing each field's mean, its relative L2 difference
was `8.118e-3`.

The MLP reduced the `nut` relative L2 error to `1.881e-1`, but its output
constraint remains incomplete. It produced negative `nut` in 2.95 percent of
cells, with a minimum of `-8.077e-7`, because the offline
`maximum(prediction, 0)` operation was not part of the exported model. Its
mean `nut` was `2.489e-6` and its maximum was `1.043e-4`. The active-region
99th-percentile relative error was `5.121`, worse than either tree model even
though its global norms and resulting `U` and `p` fields were substantially
better. A production gate therefore requires the nonnegative output rule to
be represented inside the model or artifact contract and retested in the
solver.

### Timing

| Path | Artifact payload | Started (Europe/Helsinki) | Host | OpenFOAM ExecutionTime [s] | OpenFOAM ClockTime [s] | FoamNordic total [s] | Orchestration [s] |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Stock Smagorinsky | n/a | 14:57:35 | `roihu-cpu-login1` | 44.96 | 46 | n/a | n/a |
| Mathematical `nutFjord` | 1.9 KiB | 14:48:55 | `rc5183` | 52.15 | 55 | 56 | 1 |
| Learned Joblib `VotingRegressor` | about 186 MiB | 16:06:00 | `rc5183` | 54.32 | 665 | 698 | 33 |
| Learned Joblib `ExtraTrees` | about 180 MiB | 16:46:15 | `rc5183` | 53.97 | 105 | 108 | 3 |
| Learned Equinox MLP | 32.6 KiB | 17:14:30 | `rc5183` | 52.55 | 56 | 59 | 3 |

The stock run used a login node, so it is a numerical reference and only an
indicative timing reference. The four FoamNordic runs used the same compute
node and are directly comparable. Relative to mathematical `nutFjord`, Voting
added 610 seconds of OpenFOAM wall time over 2,000 exchanges, approximately
305 ms per exchange. Extra Trees added 50 seconds, approximately 25 ms per
exchange. The Equinox MLP added one second at the log's whole-second
resolution, approximately 0.5 ms per exchange. OpenFOAM CPU `ExecutionTime`
remained within 2.2 seconds across all four paths, so the additional wall time
is resident model evaluation rather than additional solver work.

Removing exact K-nearest-neighbor inference reduced solver wall time from 665
to 105 seconds and total time from 698 to 108 seconds. Extra Trees was 6.3
times faster across the solver trajectory and reduced model-wait overhead by
about a factor of 12, despite reducing the artifact by only about 6 MiB. Both
trajectories completed all 2,000 exchanges and wrote `t=0.02`; neither
stalled. The comparison identifies the K-nearest-neighbor component, not SHM
transport or lifecycle orchestration, as the dominant bottleneck in the
composite model. A production comparison must retain the held-out trajectory
criterion while replacing or reducing inference components that cannot
satisfy the per-step latency budget.

The Equinox trajectory completed the same 2,000 exchanges in 56 seconds of
solver wall time, effectively matching the 55-second mathematical closure and
finishing 1.9 times faster than Extra Trees and 11.9 times faster than Voting.
Together with its 32.6 KiB payload and improved global field errors, this
makes the vectorized Equinox model the strongest learned runtime candidate in
this gate. Its next acceptance step is output-constraint parity, not transport
or lifecycle optimization.
