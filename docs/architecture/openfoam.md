# OpenFOAM integration
FoamNordic integrates through a small ABI-specific library built against the active OpenFOAM environment. The wheel supplies the SDK and native runtime; foamnordic build compiles the integration for the selected ABI.

## Field programs

A Transform reads and writes registered fields at a declared solver stage. A Closure enters an equation-level adapter owned by the solver. Both use the same transport, FNOM, resource, and failure contracts.

Supported stored-field layouts include scalar, vector, symmetric tensor, tensor, and explicitly registered surface fields. Derived expressions such as grad, div, curl, laplacian, magnitude, symmetric and deviatoric parts, contractions, and checked scalar algebra are evaluated by OpenFOAM from the current mesh and field registry.

### Binding rules

- Logical model port names are separate from OpenFOAM field names.
- Shapes and dimensions are validated before launch and again at exchange.
- Field storage is resolved for every invocation; mesh-owned pointers are not retained across topology or storage changes.
- Each solver adapter owns a monotonic topology generation. A changed mesh instance, cell/face/point count, topology-changing transition, or explicit mesh-update callback advances it; requests and responses must retain that exact generation.
- Patch selection is explicit and does not imply whole-field correction.
- Multiple writers may not target the same field at the same solver stage.

## Closures

Closure adapters fix the solver-owned outputs and equation semantics. Their model inputs remain explicit and may use any compatible stored or derived field expression.

### nutFjord

`nutFjord` is the algebraic eddy-viscosity adapter. It publishes one solver-owned field,

$$
\nu_t = \mathcal{M}(\mathbf{x}),
$$

where the model inputs $\mathbf{x}$ are declared by the user. The default binding is `grad(U), delta`, but an explicit FNOM contract may instead use any supported stored or derived field expression. `nutFjord` fixes only the `nut` output and its scalar layout. OpenFOAM retains ownership of field storage, boundary correction, and momentum-equation coupling.

### kEqnFjord

`kEqnFjord` retains the OpenFOAM transport equation for subgrid kinetic energy and delegates three closure terms:

$$
(\nu_t, P_k, C_\varepsilon) = \mathcal{M}(\mathbf{x}).
$$

These outputs are fixed as `nut`, `kProduction`, and `kDissipationCoeff`. The declared inputs remain selectable, although a model representing a conventional one-equation closure normally includes the current `k` field. `k` is read from solver-owned storage at every closure invocation, including repeated correctors at the same physical time; it is not a detached value from the preceding time step.

The resulting source and sink terms enter the solver-owned equation as

$$
\frac{\partial (\alpha \rho k)}{\partial t}
+ \nabla \cdot (\alpha \rho \mathbf{u} k)
- \nabla \cdot (\alpha \rho D_k \nabla k)
= \alpha \rho P_k
- \alpha \rho C_\varepsilon k
- \frac{2}{3}\alpha\rho(\nabla\cdot\mathbf{u})k
+ S_k.
$$

The adapter solves and bounds `k`, then invokes the closure again with the updated field. Model inputs may therefore change without changing the equation adapter, while the output names, layouts, dimensions, and equation semantics remain fixed.

### progressRateFjord

`progressRateFjord` exposes an equation hook for solvers that already own a transported progress variable and its reaction-rate field. Its input contract is deliberately open:

$$
\omega_c = \mathcal{M}(\mathbf{x}).
$$

The consumer may bind `progress`, `variance`, `temperature`, arbitrary stored fields, or compatible derived expressions as $\mathbf{x}$. The hook fixes one scalar logical output, `omega_c`, and rejects any other output contract. It does not transport progress, add a source to an equation, define heat release, or update thermochemical state. Those operations remain the responsibility of the consuming solver library, which also registers the hook in its own model hierarchy.

Model input names remain logical contract keys. A model exported with `c`, `c_var`, and `T`, for example, can map those keys to the hook's `progress`, `variance`, and `temperature` bindings through `inputExpressions`; no input vocabulary is imposed by `progressRateFjord`.

## Numerics

Derived operations require entries in fvSchemes. The default convenient policy keeps an exact user entry or a usable default and inserts only a missing requirement. Every insertion is recorded in numerics-changes.tsv. Strict numerics rejects the missing requirement instead.

## Atomic publication

All outputs are staged and validated before commit. A bad shape, conversion, non-finite value, or overlapping destination rejects the whole exchange. Boundary correction remains solver-owned unless an adapter explicitly defines another contract.

## External adapters

An external solver can include the installed headers, link the ABI-matched integration library, and register its equation hook without a FoamNordic source checkout. Directory discovery is available through:

    foamnordic dir --include
    foamnordic dir --openfoam-library

See the externalNutFjord example for the minimal build and registration shape.
