# Combustion contract

This document defines the native boundary for FoamNordic progress-variable
combustion. The reaction-rate source adapter, two-program coordinator,
pre-integrated FNOM manifold dispatch, native progress-equation source matrix,
and a deliberately narrow reference solver are implemented. Production solver
families may replace the reference transport equations while retaining the
same semantic and lifecycle contracts.

## Scientific ownership

The solver owns transported variables, equation assembly, relaxation,
constraints, boundary conditions, and thermodynamics. FoamNordic supplies two
replaceable closures:

| Closure | Semantic inputs | Semantic outputs |
| --- | --- | --- |
| Reaction rate | `progress`, `variance`, `temperature`, optional conditioning | `reaction_rate` |
| Manifold | `progress`, `variance`, optional conditioning | species family, enthalpy, optional thermochemical fields |

Semantic ports are stable; case field names are bindings. A solver may bind
`progress` to `c_tilde`, `Chi`, or another scalar without changing the runtime.

Surveyed OpenFOAM combustion implementations consistently reinforce this
boundary:

- progress, mixture-fraction, variance, enthalpy, and scalar-dissipation
  equations remain solver responsibilities;
- reaction-rate models expose narrow correction, equation-source, and heat
  release entry points;
- one manifold coordinate is reused for species and thermophysical outputs;
- boundary lookup and thermodynamic correction belong to the solver adapter,
  not the table backend;
- spatially uneven detailed chemistry may require work redistribution, but
  that scheduling remains separate from the closure contract.

FoamNordic therefore does not copy an FGM, FPV, flamelet, or detailed-chemistry
solver. It preserves their stable coupling pattern while leaving equations and
thermodynamic state in the target application.

## Initial numerical policy

- Progress is normalized to `[0, 1]`.
- Variance is non-negative and no larger than
  `progress*(1-progress)` unless a concrete model declares another domain.
- The reaction-rate source is lagged by one declared outer correction.
- The beta-FDF table is integrated offline and stored as an FNOM artifact.
- Out-of-domain manifold coordinates are clipped by default; an `error` policy
  is available for validation runs.
- Species and thermochemical fields are updated as one manifold transaction,
  followed by exactly one native `thermo.correct()` call.

The first native adapter defines `reaction_rate` as a signed, positive
right-hand-side production field. A consuming model returns negative values;
there is no hidden sign flip. The coupling policy makes its basis explicit:
`volumetric_mass` means `[kg/m3/s]`, while `specific` means `[1/s]` and is
multiplied by solver-owned density exactly once during equation assembly.
OpenFOAM rejects a field whose dimensions disagree with the selected basis.

## Required correction order

1. Prime the source before the first scalar solve.
2. Solve progress and variance using the declared source treatment.
3. Evaluate the reaction-rate closure from the updated state.
4. Evaluate the manifold from the updated moments.
5. Apply and bound all returned fields atomically.
6. Correct boundaries and thermodynamics.
7. Continue pressure-density coupling.

This order must be encoded at a solver-native correction site. A generic
time-step `Transform` is not a substitute for an equation-level combustion
adapter.

## Parallel execution contract

Each request carries rank and local-cell identity. Implementations may batch,
cache, balance, or reorder work, but each solution must be scattered back by
identity rather than arrival order. Timing should distinguish pack, transfer,
evaluate, table lookup, scatter, and thermodynamic correction. There is no
Python per-cell loop in the native hot path.

## Reference modes and acceptance gates

A concrete solver integration must retain a pure OpenFOAM or table-only mode.
Acceptance proceeds through pointwise table fixtures, a frozen-field adapter
comparison, a short coupled trajectory, and only then a representative MPI
case. Reference implementations are scientific oracles, not source trees to
copy wholesale; compatibility and corrected-physics baselines must remain
separate.

The copyable guarded files live in
`src/foamnordic/template/openfoam/combustion-model/`.

## Implemented coordinator

`progressVariableFjord` is registered for both `psiReactionThermo` and
`rhoReactionThermo`. Its first `ClosureHook` writes exactly one dimensioned
reaction-rate field. Its second hook updates the expanded manifold outputs as
one exchange. Only after both invocations succeed does it call
`thermo.correct()`, when enabled. The model rejects unsupported source and
correction policies, missing semantic inputs, missing or non-scalar output
fields, a manifold that overwrites the source, and source-dimension mismatch.

The Python compiler lowers `ProgressVariable` to two ordinary resident model
programs and gives each an independent Unix session. `Y_*`-style field
families are expanded against the initial registry before launch. The shared
Longship lifecycle starts and terminates the two workers together; there is no
per-cell Python loop.

The solver still owns steps 1, 2, 5, and 7 of the correction order. It
assembles the source through the standard
`transport == combustion->R(progress)` boundary and calls
`combustion->correct()` exactly once at the agreed outer-corrector boundary.
`R()` returns a positive explicit source only for the configured progress
field; unrelated scalar and species equations receive a zero matrix.

Both `reactionRateFjord` and `progressVariableFjord` accept the same immutable
`reactionRateBasis` dictionary contract. The default `volumetricMass` path
preserves the original behavior. The `specific` path reads `thermo.rho()` and
converts the model output to a volumetric source immediately before matrix
assembly; density must therefore not be baked into the model artifact.

## Reference solver

`foamnordicProgressVariableFoam` is the first solver-integrated acceptance
path. `foamnordic build` compiles it beside the ABI-matched OpenFOAM adapter in
the managed runtime `bin/` directory, and coupled launch adds that directory to
the solver environment automatically.

The reference solver uses `psiReactionThermo` and a transient PIMPLE loop. It
solves configurable progress and variance fields, invokes
`combustion->correct()` after those solves, and then continues pressure-density
coupling. Progress diffusion uses `muEff/progressSchmidt`; variance is a
transported LES moment using `muEff/varianceSchmidt`, bounded by
`0 <= variance <= progress*(1-progress)`. Its SGS production is
`2*(mu_t/Sc_t)*|grad(progress)|^2` and its modeled dissipation is
`Cchi*(mu_t/Sc_t)*variance/delta^2`, matching the transported-variance form
used by the flameletFoam reference after the molecular-gradient contribution
cancels. The portable solver uses `delta=cubeRootVol`; the field binding,
Schmidt numbers, and `Cchi` live in
`constant/progressVariableTransportProperties`.

This scope is intentional. The solver does not copy a fork-specific turbulence
class or scalar-dissipation field, does not support local-time stepping, and does
not solve stock species or energy equations. Manifold outputs own the selected
species transaction. A production FPV thermodynamics adapter is still needed
before manifold enthalpy and temperature can replace a solver family's native
energy ownership.

The reduced gate at `tools/openfoam/testProgressVariableCombustion.py` was run
with OpenFOAM.com v2606 on the 4,000-cell counter-flow flame mesh for two
transport steps. Both Joblib programs used native shared-memory exchanges; the
progress equation, variance equation, reaction source, and two-field manifold
transaction completed in one Longship lifecycle. Final ranges were
`c = 0.245614 .. 0.260865` and
`omega_c = 0.0147827 .. 0.0150877`; the maximum `|CH4 + CO2 - 1|` was exactly
zero. The OpenFOAM solver execution time was 0.05 seconds; lifecycle startup
and shutdown took about two seconds on the validation machine.

## Implemented source boundary

`reactionRateFjord` is a modern OpenFOAM `ThermoCombustion` runtime model for
`psiReactionThermo` and `rhoReactionThermo`. It performs one native Fjord
exchange from the required semantic inputs `progress`, `variance`, and
`temperature` to one dimensioned `volScalarField`. Additional conditioning
inputs are permitted. A custom progress-variable solver may register and own
that field. A stock reacting solver may instead declare
`autoCreateReactionRate true`, in which case the adapter registers and owns the
source without requiring a copied solver. Field identity, ownership mode, and
dimensions are immutable during a run, and mismatches fail before the closure
session starts.

The model is only a reaction-rate producer. Its `R(progress)` method exposes
the source through the ordinary OpenFOAM combustion interface, while
`R(otherField)` and heat release return zero. It never corrects
thermodynamics. This keeps equation assembly and correction order under solver
ownership and prevents accidental sources in unrelated species equations.

`tools/openfoam/reactionRateReactingFoam.py` validates this adapter in the
unmodified stock `reactingFoam` executable. On the 4,000-cell counter-flow
case, both the `[kg/m3/s]` `volumetric_mass` path and the `[1/s]` `specific`
path completed with one resident Joblib model and a positive finite source;
their local lifecycle times were 2.55 and 2.38 seconds, respectively. This is
an integration gate. The separate source probe below supplies the exact
`V*omega` and `V*rho*omega` numerical assertions.

## Solver-native source gate

`foamNordic::combustion::explicitSource()` is the shared C++ boundary used by
both combustion models. It constructs an `fvScalarMatrix` whose dimensions are
the configured volumetric-source dimensions multiplied by cell volume and
adds the source with OpenFOAM's positive right-hand-side convention. The
solver combines that matrix with its own transport matrix, which supplies the
final dimensional compatibility check.

`tools/openfoam/progressVariableSourceProbe` loads arbitrary solver-owned
`c_tilde` and `omega_c` fields, constructs the matrix, and compares every cell
against `V*omega_c`. It also constructs independent in-memory density and
specific-rate fields and compares every cell against `V*rho*omega`. The gate
was compiled and run with OpenFOAM.com v2606 on a 4,000-cell counter-flow mesh;
the maximum volumetric-source error was exactly zero and the maximum
specific-source error was `5.17e-26`. It tests native sign, density and volume
scaling, and dimensions independently of the learned model.

The single-cell reference module also carries a three-step lagged trajectory
gate for a signed volumetric mass source. It primes the source at the initial
state, advances progress with `dt*omega/rho`, evaluates the next source from
the solved state, and finally evaluates the beta-FDF manifold. Its frozen
progress sequence is `0.36, 0.488, 0.5904`; each linear table expectation
matches the updated progress to numerical precision. This locks call ordering
independently of transport discretization.

## Single-cell reference oracle

`python/foamnordic/combustion/reference.py` provides a deliberately slow,
single-cell beta-FDF oracle for acceptance tests. It handles zero variance as
a delta distribution, maximum admissible variance as an endpoint mixture, and
ordinary moments as a beta distribution. Piecewise-linear table segments are
integrated using regularized incomplete-beta moments, without Monte Carlo
sampling or runtime Python quadrature.

The frozen fixture at
`python/tests/fixtures/combustion/beta_fdf_single_cell.json` covers both
endpoints, a delta state, a uniform beta distribution, and the maximum-variance
limit. This oracle validates generated manifold artifacts offline; it must not
be called from an OpenFOAM cell loop.

The call-order oracle is frozen separately at
`python/tests/fixtures/combustion/progress_variable_trajectory.json`. It fixes
the initial state, lagged volumetric source, three solved progress states, next
sources, and manifold values. Tests consume the fixture rather than embedding
those expected numbers in Python, so a source-treatment or correction-order
change requires an explicit oracle revision.

## Target-application compatibility boundary

A representative target application establishes the intended scientific shape
of a production artifact: inputs are `c_tilde`, `c_var`, and `T_tilde`, and the
learned target `omega_c_chem_tilde` is a specific progress rate in `[1/s]`.
Such an artifact must declare
`CouplingPolicy(reaction_rate_basis="specific")`; FoamNordic then performs the
sole `rho*omega` conversion in native OpenFOAM code.

Its beta-FDF notebook uses the normalized variance coordinate
`g = c_var/(c_tilde*(1-c_tilde))`, including explicit delta and endpoint
limits. That coordinate transformation belongs in the artifact builder and
metadata, not as an implicit field rename. The
operating condition (for example equivalence ratio and pressure), progress
variable definition, species convention, and training-domain distance also
need durable metadata. Distance/region diagnostics should be observations;
they must not become extra transported fields merely to accommodate a
particular estimator.

The current Joblib resident accepts importable estimators operating on dense
arrays. Notebook-local wrapper classes such as distance-blended or
multi-region regressors must be moved into an importable module or exported
through a stable adapter before packaging. This is an artifact portability
constraint, not a reason to specialize the OpenFOAM solver or Fjord protocol.
