# OpenFOAM model adapters

This directory contains equation-level OpenFOAM model adapters. An adapter
belongs here only when FoamNordic must enter a model's native correction or
source-evaluation method. It should select solver-owned fields and invoke the
shared `ClosureHook`; transport, inference, scaling, observation, and lifecycle
code remain in the common FoamNordic runtime.

The current adapters are LES closures:

- `nutFjord` replaces the modeled eddy-viscosity evaluation.
- `kEqnFjord` replaces the one-equation SGS closure terms.

Generic field programs do not require a turbulence-model adapter.
`foamNordicExchange` can transform or observe registered fields in laminar,
RAS, and LES cases at the supported solver stages. Do not add parallel
`laminar`, `RAS`, or `LES` copies merely to host the same field-exchange code.

Introduce category subdirectories only when their native contracts are real
and distinct. Likely future examples are a RAS adapter that enters a modeled
transport equation, and combustion adapters that evaluate reaction-rate or
table-coupled source terms. At that point, move related adapters together and
update `makeClosureModels.C` and `Make/files` in the same change.

Every adapter must remain thin and must have a solver-integrated acceptance
case. A successful generic `Transform` smoke test is evidence for the shared
field bridge, not evidence that a new equation-level model adapter is needed.
