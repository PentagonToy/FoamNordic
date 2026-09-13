# OpenFOAM equation adapters

These adapters enter an OpenFOAM model at its native correction or source-evaluation boundary. They select solver-owned fields and invoke the shared `ClosureHook`; OpenFOAM retains equation assembly and field ownership, FoamNordic retains transport and lifecycle, and BindNordic retains model execution.

> **Fjord model contract:** inputs are freely selectable expressions, while outputs are fixed by the adapter. A mathematical operator and a trained model may use different inputs, but they must return the same adapter-owned outputs.

| Adapter | Native boundary | Fixed outputs |
| --- | --- | --- |
| `nutFjord` | LES eddy-viscosity correction | `nut` |
| `kEqnFjord` | One-equation SGS correction | `k`, `nut` |
| `progressRateFjord` | External progress-variable reaction-rate hook | `omega_c` |

`nutFjord` and `kEqnFjord` are registered for incompressible and compressible LES models. A fully symbolic `nutFjord` function is compiled once and evaluated directly inside OpenFOAM; model-backed and otherwise unsupported functions retain the resident ModelHost path. `progressRateFjord` is an unregistered hook: a solver project such as FjordFoam registers it in its own model hierarchy and decides how `omega_c` enters its equations.

`kEqnFjord` accepts either a model that directly updates `k` and `nut`, or a mathematical `Operator.function` containing one `Field.transport()` call. The latter is compiled once into OpenFOAM expressions and a native finite-volume solve; it does not launch a ModelHost or exchange fields during correctors. The earlier `nut`, `kProduction`, and `kDissipationCoeff` resident contract remains accepted for compatibility.

Generic field programs do not require a model adapter. `foamNordicExchange` transforms or observes registered fields in laminar, RAS, and LES cases at supported solver stages.

## Extension rule

Choose an adapter by its equation entry point, not by `simulationType`. Domain-specific equations, mesh policies, reaction models and transported scalars remain in the solver project. New adapters must remain thin, declare fixed outputs and include a solver-integrated acceptance case. The starting scaffold is in [`tools/templates/openfoam/scaffolds/model-adapter/`](../../../../tools/templates/openfoam/scaffolds/model-adapter/README.md).
