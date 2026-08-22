# Progress-variable combustion contract

This document defines the provisional native boundary for FoamNordic
progress-variable combustion. It is a design contract and guarded scaffold,
not a claim that a combustion solver is currently implemented.

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

The source units and sign are not globally assumed. A concrete adapter must
declare whether its equation consumes a volumetric mass source, a normalized
progress rate, or another dimensional form and reject an incompatible model
manifest.

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
