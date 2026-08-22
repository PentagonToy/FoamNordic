# Progress-variable combustion contract

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
there is no hidden sign flip. Units remain explicit: a concrete case declares
whether the field is a volumetric mass source, normalized progress rate, or
another dimensional form, and OpenFOAM rejects an incompatible equation.

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
case. Legacy implementations are scientific oracles, not source trees to copy
wholesale; compatibility and corrected-physics baselines must remain separate.

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

## Reference solver

`foamnordicProgressVariableFoam` is the first solver-integrated acceptance
path. `foamnordic build` compiles it beside the ABI-matched OpenFOAM adapter in
the managed runtime `bin/` directory, and coupled launch adds that directory to
the solver environment automatically.

The reference solver uses `psiReactionThermo` and a transient PIMPLE loop. It
solves configurable progress and variance fields, invokes
`combustion->correct()` after those solves, and then continues pressure-density
coupling. Progress diffusion uses `muEff/progressSchmidt`; variance is a
portable passive-moment equation using `muEff/varianceSchmidt`, bounded by
`0 <= variance <= progress*(1-progress)`. The field binding and Schmidt numbers
live in `constant/progressVariableTransportProperties`.

This scope is intentional. The solver does not copy fork-specific variance
production/destruction methods, does not support local-time stepping, and does
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
`temperature` to one solver-owned `volScalarField`. Additional conditioning
inputs are permitted. Field identity and dimensions are immutable during a
run, and mismatches fail before the closure session starts.

The model is only a reaction-rate producer. Its `R(progress)` method exposes
the source through the ordinary OpenFOAM combustion interface, while
`R(otherField)` and heat release return zero. It never corrects
thermodynamics. This keeps equation assembly and correction order under solver
ownership and prevents accidental sources in unrelated species equations.

## Solver-native source gate

`foamNordic::combustion::explicitSource()` is the shared C++ boundary used by
both combustion models. It constructs an `fvScalarMatrix` whose dimensions are
the configured volumetric-source dimensions multiplied by cell volume and
adds the source with OpenFOAM's positive right-hand-side convention. The
solver combines that matrix with its own transport matrix, which supplies the
final dimensional compatibility check.

`tools/openfoam/progressVariableSourceProbe` loads arbitrary solver-owned
`c_tilde` and `omega_c` fields, constructs the matrix, and compares every cell
against `V*omega_c`. The gate was compiled and run with OpenFOAM.com v2606 on a
16,384-cell cavity mesh; the maximum source error was exactly zero. It tests
native sign, volume scaling, and dimensions without claiming combustion
physics for the cavity.

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
limit. This oracle is the first comparison target for a future native/FNOM
manifold backend; it must not be called from an OpenFOAM cell loop.
