# Numerical APIs

`fno.Math` provides backend-neutral array algebra, while `fno.Field` describes OpenFOAM fields and finite-volume operations. NumPy arrays remain in NumPy, JAX arrays and tracers remain in JAX, and symbolic field expressions are evaluated by OpenFOAM at the declared solver boundary.

## Supported operations

| API | Operations | OpenFOAM native graph |
|---|---|:---:|
| `fno.Math` scalar algebra | `abs`, `sqrt`, `square`, `maximum`, `minimum`, `clip`, positive dyadic powers | Yes |
| `fno.Math` selection | `where(field > value, a, b)` | Yes |
| `fno.Math` tensor algebra | `mag`, `symm`, `dev`, `trace`, `outer`, `dot`, `ddot`, `matmul` | Yes |
| `fno.Math.einsum` | `"...ii->..."`, `"...ij,...ij->..."` | Yes |
| `fno.Math` general element-wise | `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tan`, `tanh` | Resident |
| `fno.Math` reductions and layouts | `sum`, `mean`, `min`, `max`, `reshape`, `transpose`, `stack`, `concatenate`, other `einsum` patterns | Resident |
| `fno.Field` declarations | `Field(name)`, `Field.delta` | Yes |
| `fno.Field` geometry | `Field.coordinate` | Resident |
| `fno.Field` finite-volume expressions | `Field.grad`, `Field.div`, `Field.laplacian`, `Field.curl` | Yes |
| `fno.Field` solver control | `Field.transport` | Yes, `kEqnFjord` |
| `fno.Context` runtime values | `Context.delta_time` | Yes |

`Yes` means that a fully symbolic function can execute inside OpenFOAM without a ModelHost or field exchange. `Resident` means that the operation remains available to NumPy or JAX functions evaluated by the ModelHost. Mixed functions use the resident path rather than splitting one function between execution lanes.

## `Math` reference

All `Math` functions return the native result type of their operands: NumPy input produces NumPy output, JAX input remains in JAX, and a symbolic `FieldExpression` produces another expression. Arguments named `axis`, `axes`, `shape` and `subscripts` follow NumPy conventions on the resident path.

| Function | Signature | Default or constraint |
| --- | --- | --- |
| Unary arithmetic | `abs(x)`, `sqrt(x)`, `square(x)` | `sqrt` requires a non-negative numerical value or scalar field expression |
| Exponential and logarithmic | `exp(x)`, `expm1(x)`, `log(x)`, `log1p(x)` | Resident path |
| Trigonometric | `sin(x)`, `cos(x)`, `tan(x)`, `tanh(x)` | Resident path |
| Bounds | `maximum(a, b)`, `minimum(a, b)`, `clip(x, minimum=None, maximum=None)` | At least one `clip` bound is required |
| Selection | `where(condition, when_true, when_false)` | Native graph supports scalar field comparison conditions |
| Reductions | `sum(x, axis=None, keepdims=False)`, `mean(...)`, `min(...)`, `max(...)` | Resident path |
| Layout | `reshape(x, shape)`, `transpose(x, axes=None)` | Resident path |
| Joining | `stack(values, axis=0)`, `concatenate(values, axis=0)` | Non-empty iterable |
| Tensor contraction | `einsum(subscripts, *operands)`, `matmul(a, b)`, `dot(a, b)`, `ddot(a, b)` | Native patterns are listed above |
| Tensor construction | `outer(a, b)` | Vector × vector → tensor |
| Tensor properties | `trace(A)`, `mag(x)`, `symm(A)`, `dev(A)` | Component-compatible operands |

`fno.Math.outer(a, b)` maps two vector fields to a tensor field, and `fno.Math.trace(A)` maps a tensor field to a scalar field. `fno.Context.delta_time()` reads the active OpenFOAM time-step size at the invocation boundary.

Native field graphs normalise the common tensor forms `matmul(A, B)`, `einsum("...ii->...", A)` and `einsum("...ij,...ij->...", A, B)` to OpenFOAM contractions. Positive dyadic powers such as `x**2`, `x**1.5`, `x**1.25` and `x**2.5`, together with scalar `where(x > value, a, b)`, remain native. A positive dyadic power has an exponent representable as a positive integer divided by a power of two. Other powers, `einsum` patterns and arbitrary Python control flow retain the resident path.

## Mathematical operators

The same mathematical function can operate on ordinary arrays for local checks or on symbolic `FieldExpression` values during closure declaration.

```python
# Define the Smagorinsky function

def smagorinsky_function(velocity_grad, delta, C_k=0.094, C_e=1.048):
    strain_rate = fno.Math.symm(velocity_grad)
    strain_rate_trace = fno.Math.trace(strain_rate)
    dev_strain_rate = fno.Math.dev(strain_rate)

    coefficient_a = C_e / delta
    coefficient_b = (2.0 / 3.0) * strain_rate_trace
    coefficient_c = (
        2.0
        * C_k
        * delta
        * fno.Math.ddot(dev_strain_rate, strain_rate)
    )

    discriminant = coefficient_b**2 + 4.0 * coefficient_a * coefficient_c
    sqrt_k = (
        -coefficient_b
        + fno.Math.sqrt(fno.Math.maximum(discriminant, 0.0))
    ) / (2.0 * coefficient_a)
    sqrt_k = fno.Math.maximum(sqrt_k, 0.0)

    return C_k * delta * sqrt_k
```

Numerical safeguards belong to the function or trained model that defines the closure. FoamNordic does not silently clip a resident model output after inference.

## OpenFOAM field expressions

Stored fields and derived fields use the same input declaration.

```python
# Define OpenFOAM closure inputs

inputs = {
    "velocity_grad": fno.Field.grad("U"),
    "convection": fno.Field.div("phi", "U"),
    "thermal_diffusion": fno.Field.laplacian("alphaEff", "T"),
    "vorticity": fno.Field.curl("U"),
    "delta": fno.Field.delta(),
}
```

Field expressions compose without manually constructing OpenFOAM expression strings.

```python
# Define the strain-rate magnitude

velocity_grad = fno.Field.grad("U")
strain = fno.Math.symm(velocity_grad)
strain_rate = fno.Math.sqrt(2.0 * fno.Math.ddot(strain, strain))
```

Field expressions support matching-field addition and subtraction. Multiplication accepts a scalar field or finite dimensionless scalar on either side; division accepts a scalar field or finite dimensionless scalar as its denominator. `sqrt` requires a scalar field. `Field.laplacian(diffusivity, field)` accepts a scalar field expression as its diffusivity.

### `Field` and `Context` reference

| API | Signature | Result or constraint |
| --- | --- | --- |
| Stored field | `fno.Field(name)` | Field named by a non-empty OpenFOAM word |
| Explicit stored field | `fno.Field.field(name)` | Equivalent to `fno.Field(name)` |
| Boundary patch | `fno.Field.patch(name, patch_name)` | One named patch of a stored field |
| Field family | `fno.Field.fields(pattern)` | Wildcard selection containing at least one `*` |
| Gradient | `fno.Field.grad(field)` | Gradient expression |
| Filter width | `fno.Field.delta()` | Active LES filter-width field |
| Coordinate | `fno.Field.coordinate(axis)` | Coordinate scalar; `axis` is `x`, `y` or `z` |
| Divergence | `fno.Field.div(field)` or `fno.Field.div(flux, field)` | Explicit divergence expression |
| Laplacian | `fno.Field.laplacian(field)` or `fno.Field.laplacian(diffusivity, field)` | Explicit Laplacian expression |
| Curl | `fno.Field.curl(field)` | Curl expression |
| Runtime timestep | `fno.Context.delta_time()` | Active OpenFOAM `deltaT` at invocation |

`Field.fields()` is output metadata rather than a finite-volume expression. It expands against the solver object registry and fails if the pattern matches no fields. Patch expressions identify boundary data explicitly and remain subject to the selected adapter contract.

```python
fno.Field.transport(value, *, diffusivity, source, implicit_sink,
                    lower_bound=0.0)
```

| Parameter | Type | Default | Constraint |
| --- | --- | --- | --- |
| `value` | field name or `FieldExpression` | required | Scalar transported field |
| `diffusivity` | field name or `FieldExpression` | required | Scalar effective diffusivity |
| `source` | field name or `FieldExpression` | required | Explicit source term |
| `implicit_sink` | field name or `FieldExpression` | required | Scalar implicit sink coefficient |
| `lower_bound` | `float` | `0.0` | Finite post-solve lower bound |

The finite-volume operators use the copied case's `system/fvSchemes`. FoamNordic preserves an exact user-supplied scheme or a usable section default. In convenient mode it adds an exact fallback only when neither is available; strict mode rejects the missing scheme. Current fallbacks are `Gauss linear` for gradient, curl and divergence expressions and `Gauss linear corrected` for Laplacians. The source case is never changed.

## Native algebra

A symbolic `Operator.function` can run directly in OpenFOAM when every operation is supported and it uses complete fields. Functions that use patches or unsupported Python operations run through ModelHost instead. Native closures and transforms support `Observe` and `profile_every` without exchanging fields.

```python
# Define the Smagorinsky closure

closure = fno.Closure(
    name="nutFjord",
    operator=fno.Operator.function(smagorinsky_function),
    inputs={
        "velocity_grad": fno.Field.grad("U"),
        "delta": fno.Field.delta(),
    },
    outputs={
        "nut": fno.Field("nut"),
    },
)
```

Functions containing Python control flow, NumPy, JAX or model evaluation remain on the resident path. Their OpenFOAM input expressions are still evaluated before packing, so derived fields do not move to Python merely because the model is resident.

## Native transport

`Field.transport()` marks one implicit scalar finite-volume solve inside a mathematical `Operator.function`. The graph is traced during declaration and executed by `kEqnFjord` without a ModelHost in the corrector loop.

```python
# Define the k-equation function

def keqn_function(k, velocity_grad, nu, delta, C_k=0.094, C_e=1.048):
    k_positive = fno.Math.maximum(k, 1.0e-10)
    sqrt_k = fno.Math.sqrt(k_positive)
    strain = fno.Math.dev(2.0 * fno.Math.symm(velocity_grad))

    nut = C_k * delta * sqrt_k
    production = nut * fno.Math.ddot(velocity_grad, strain)
    dissipation = C_e * sqrt_k / delta

    k_new = fno.Field.transport(
        value=k,
        diffusivity=nu + nut,
        source=production,
        implicit_sink=dissipation,
        lower_bound=1.0e-10,
    )

    return {
        "k": k_new,
        "nut": C_k * delta * fno.Math.sqrt(k_new),
    }


# Define the k-equation closure

closure = fno.Closure(
    name="kEqnFjord",
    operator=fno.Operator.function(keqn_function),
    inputs={
        "k": fno.Field("k"),
        "velocity_grad": fno.Field.grad("U"),
        "nu": fno.Field("nu"),
        "delta": fno.Field.delta(),
    },
    outputs={
        "k": fno.Field("k"),
        "nut": fno.Field("nut"),
    },
)
```

The current native transport contract permits one scalar solve and fixes the `kEqnFjord` outputs as `k` and `nut`. Use the value returned by `Field.transport()` for post-solve calculations; mixing it with the original `k` in one output is rejected. `Observe` samples the requested OpenFOAM fields after completed corrections, and `profile_every` measures native algebra, matrix assembly, solving and field publication.

## Random keys

FoamNordic uses immutable, backend-neutral keys for stochastic closures and transforms.

```python
key = fno.Random.key(42, scope="global")


def perturb(velocity, *, key):
    scale_key, noise_key = fno.Random.split(key, 2)
    scale = fno.Random.uniform(scale_key, low=0.995, high=1.005)
    noise = fno.Random.normal(noise_key, shape=velocity.shape, std=1.0e-6)
    return {"velocity": velocity * scale + noise}
```

Keys are never consumed. Reusing a key repeats a draw; split it when one function needs independent draws. The available distributions are `uniform`, `normal`, `integers` and `bernoulli`. `scope="global"` gives every solver rank the same invocation key, while `scope="rank"` also folds in the negotiated rank. `fno.Random.to_jax(key)` materialises a JAX key. NumPy and JAX are reproducible within their backend but do not promise bit-identical samples across backends.

| API | Principal arguments | Defaults or constraint |
| --- | --- | --- |
| `Random.key` | `seed`, `scope` | `scope="global"`; scope is `global` or `rank` |
| `Random.split` | `key`, `count` | `count=2`; positive integer |
| `Random.uniform` | `key`, `low`, `high`, `shape` | `low=0.0`, `high=1.0`, `shape=None` |
| `Random.normal` | `key`, `mean`, `std`, `shape` | `mean=0.0`, `std=1.0`, `shape=None` |
| `Random.integers` | `key`, `low`, `high`, `shape` | `high=None`, `shape=None`; follows NumPy half-open interval semantics |
| `Random.bernoulli` | `key`, `probability`, `shape` | `probability=0.5`, `shape=None`; probability lies in `[0, 1]` |
| `Random.to_jax` | `key` | Returns a JAX PRNG key |
