# OpenFOAM model adapters

This directory contains equation-level OpenFOAM model adapters. An adapter
belongs here only when FoamNordic must enter a model's native correction or
source-evaluation method. It should select solver-owned fields and invoke the
shared `ClosureHook`; transport, inference, scaling, observation, and lifecycle
code remain in the common FoamNordic runtime.

The current adapters include LES closures:

- `nutFjord` replaces the modeled eddy-viscosity evaluation.
- `kEqnFjord` replaces the one-equation SGS closure terms.

The combustion adapters are:

- `reactionRateFjord`, a narrow one-source producer for custom solver wiring.
- `progressVariableFjord`, which invokes reaction-rate and pre-integrated
  manifold programs in order and then corrects thermodynamics once. It still
  relies on a custom solver to assemble and solve the transported progress
  and variance equations.

Generic field programs do not require a turbulence-model adapter.
`foamNordicExchange` can transform or observe registered fields in laminar,
RAS, and LES cases at the supported solver stages. Do not add parallel
`laminar`, `RAS`, or `LES` copies merely to host the same field-exchange code.

Choose an integration family by the equation entry point, not by the label in
`simulationType`:

- A time-step boundary field transform remains a generic function object.
- An algebraic or transported LES closure enters the LES model correction.
- A RAS closure enters its modeled stress or transport-equation correction;
  its state (`k`, `epsilon`, `omega`, wall treatment) is not interchangeable
  with an LES contract.
- An iLES stress or learned momentum source enters momentum-equation assembly,
  normally through a dedicated solver hook or an appropriate `fvModel`. A
  fake laminar turbulence model would put the closure at the wrong boundary.
- A combustion closure enters reaction-rate, chemistry, or scalar-source
  evaluation and owns the table fields required by that contract.

Introduce category subdirectories only when their native contracts are real
and distinct. Likely future examples are a RAS adapter that enters a modeled
transport equation, and combustion adapters that evaluate reaction-rate or
table-coupled source terms. At that point, move related adapters together and
update `makeClosureModels.C` and `Make/files` in the same change.

Every adapter must remain thin and must have a solver-integrated acceptance
case. A successful generic `Transform` smoke test is evidence for the shared
field bridge, not evidence that a new equation-level model adapter is needed.
The non-compilable starting scaffold is in
`src/foamnordic/template/openfoam/model-adapter/`.
