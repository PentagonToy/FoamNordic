# Combustion-model scaffold

This directory is a guarded scaffold for an equation-level progress-variable
combustion adapter. It is not a solver and is intentionally impossible to
compile without resolving every `@UPPER_SNAKE_CASE@` token and removing the
top-level `#error` guards.

The scaffold separates three responsibilities:

1. The solver owns transport equations and their transported fields.
2. The reaction-rate closure maps semantic inputs such as progress, variance,
   and temperature to one solver-owned source field.
3. The manifold closure maps progress-variable moments and optional
   conditioning fields to species and thermodynamic state. The adapter then
   invokes the native thermodynamic correction exactly once.

## Required native ordering

The concrete adapter must document and test this ordering before it is added
to the FoamNordic build:

1. Prime the reaction-rate source before the first transport solve.
2. Assemble and solve the progress-variable and variance equations.
3. Invoke the reaction-rate closure at the selected outer-corrector site.
4. Apply the beta-FDF manifold using the updated moments.
5. Correct species and thermodynamic field boundaries.
6. Call `thermo.correct()` once.
7. Continue pressure-density coupling using the corrected state.

An adapter may preserve a legacy one-corrector lag, but it must declare that
policy instead of acquiring it accidentally from include-file order.

## Portability rules

- Bind semantic ports to solver fields in dictionaries; never hardcode names
  such as `c`, `Chi`, `Zmix`, `T`, or `omega_c` in shared infrastructure.
- Check field class, component count, dimensions, cell count, and finite values
  before applying a returned source.
- Keep OpenFOAM equation assembly, boundary correction, and thermodynamics in
  C++. Inference, scaling, transport selection, and process lifecycle belong
  to the existing FoamNordic runtime.
- Expand species selectors against the object registry before solver launch.
  Zero matches and ambiguous aliases are errors.
- Keep beta-FDF integration offline in the first implementation. The runtime
  consumes a pre-integrated FNOM table and performs no Python cell loop.
- Preserve rank and local-cell identity through gather, evaluation, and
  scatter. Never assume that balanced work returns in input order.
- A pure OpenFOAM or table-only reference path must remain selectable for
  scientific comparison and failure fallback.

`CombustionAdapter.H.in` and `CombustionAdapter.C.in` define the native model
boundary. The equation files are insertion-point templates, not universal
transport equations. `progressVariableEqn.H.in` uses the implemented
`combustion->R(progress)` source boundary while leaving only solver-specific
transport, `fvOptions`, and bounds tokens unresolved. A solver family should
copy and resolve only the files it needs.

`reactionRateFjordProperties.in` is the exception: it is a runnable dictionary
template for the concrete `reactionRateFjord` source producer shipped in
`libfoamnordicOpenFOAM`. It requires one solver-owned scalar output and an
explicit `REACTION_RATE_DIMENSIONS` value. It does not perform manifold lookup,
species transport, heat release, or thermodynamic correction.
