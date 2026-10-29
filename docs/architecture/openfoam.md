# OpenFOAM integration

FoamNordic integrates through a small ABI-specific library built against the active OpenFOAM environment. The wheel supplies the SDK and native runtime; `foamnordic build` compiles the integration for the selected ABI.

## Integration boundaries

| Boundary | FoamNordic responsibility | Solver responsibility |
| --- | --- | --- |
| Field access | Resolve current registered storage and validate layout | Create and own fields |
| Derived inputs | Evaluate declared expressions through OpenFOAM | Supply schemes and dimensions |
| Closure hook | Invoke a validated programme and stage outputs | Define equation and correction position |
| Topology change | Carry and verify a monotonic generation | Signal storage or topology replacement |
| Publication | Validate and commit all outputs atomically | Apply boundary correction when required |

## Field programmes

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

| Adapter | Fixed outputs | Input policy | Solver-owned operation |
| --- | --- | --- | --- |
| `nutFjord` | `nut` | Any compatible stored or derived inputs | Eddy-viscosity field coupling |
| `kEqnFjord` | `k`, `nut` | Any compatible inputs; current `k` is conventional | State update or one implicit scalar transport solve |
| `progressRateFjord` | `omega_c` | Any compatible stored or derived inputs | Progress-variable source integration |

### nutFjord

`nutFjord` is the algebraic eddy-viscosity adapter. It publishes one solver-owned field,

$$
\nu_t = \mathcal{M}(\mathbf{x}),
$$

where the model inputs $\mathbf{x}$ are declared by the user. The default binding is `grad(U), delta`, but an explicit FNOM contract may instead use any supported stored or derived field expression. `nutFjord` fixes only the `nut` output and its scalar layout. OpenFOAM retains ownership of field storage, boundary correction, and momentum-equation coupling.

### kEqnFjord

`kEqnFjord` fixes the public state outputs as `k` and `nut`:

$$
(k^{n+1}, \nu_t^{n+1}) = \mathcal{M}(\mathbf{x}).
$$

Inputs remain selectable; learned one-equation closures usually include `k`. Resident models may update `k` and `nut` directly. A native `Operator.function` can instead declare one `Field.transport()` barrier. Its coefficients are lowered once; the adapter assembles the implicit equation at each correction. Both retain the same outputs.

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

Native transport solves and bounds `k`, then evaluates `nut`, without ModelHost or inter-process exchange. Mixing updated and original `k` in a post-solve output is rejected. The earlier resident `nut`, `kProduction` and `kDissipationCoeff` contract remains supported.

### progressRateFjord

`progressRateFjord` exposes an equation hook for solvers that already own a transported progress variable and its reaction-rate field. Its input contract is deliberately open:

$$
\omega_c = \mathcal{M}(\mathbf{x}).
$$

Inputs may use stored fields or compatible expressions, including progress, variance and temperature. The hook requires one scalar output, `omega_c`. The consuming solver owns transport, equation sources, heat release, thermochemical updates and hook registration.

Model input names remain logical contract keys. A model exported with `c`, `c_var`, and `T`, for example, can map those keys to the hook's `progress`, `variance`, and `temperature` bindings through `inputExpressions`; no input vocabulary is imposed by `progressRateFjord`.

## Numerics policy

Derived operations require entries in `fvSchemes`, including operations created inside a symbolic function. The default convenient policy keeps an exact user entry or a usable default and inserts only a missing requirement. Every insertion is recorded in `numerics-changes.tsv`. Strict numerics rejects the missing requirement instead.

## Atomic publication

FoamNordic validates every output before changing a field. Resident programmes reject an invalid response as one batch. Native functions likewise evaluate and validate all outputs before assigning their internal fields, then let OpenFOAM correct their boundaries. A boundary-condition failure stops the solver after the internal assignment and is not rolled back.

## External adapters

An external solver can include the installed headers, link the ABI-matched integration library, and register its equation hook without a FoamNordic source checkout. Directory discovery is available through:

```console
foamnordic dir --include
foamnordic dir --openfoam-library
```

See [`examples/openfoam/externalNutFjord`](../../examples/openfoam/externalNutFjord/README.md) for the minimal out-of-tree build and registration contract.
