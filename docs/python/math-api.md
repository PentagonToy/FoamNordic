# Math API

`fno.Math` is the backend-neutral numerical vocabulary used by closure and
transform functions. `fno.Field` separately declares OpenFOAM fields, geometry,
and finite-volume expressions. NumPy arrays remain in NumPy and JAX arrays or
tracers remain in JAX; importing FoamNordic does not eagerly import either
backend.

```python
def keqn(k, velocity_grad, filter_width, C_k=0.094, C_e=1.048):
    k_positive = fno.Math.maximum(k, 0.0)
    sqrt_k = fno.Math.sqrt(k_positive)
    nut = C_k * filter_width * sqrt_k
    strain = fno.Math.dev(2.0 * fno.Math.symm(velocity_grad))
    production = nut * fno.Math.ddot(velocity_grad, strain)
    dissipation = C_e * sqrt_k / filter_width
    return nut, production, dissipation
```

The public layer includes:

- OpenFOAM declarations: `Field(name)`, `Field.grad`, `Field.delta`,
  `Field.coordinate`, `Field.div`, `Field.laplacian`, and `Field.curl`;
- element-wise operations: `abs`, `sqrt`, `square`, `exp`, `expm1`, `log`,
  `log1p`, `sin`, `cos`, `tan`, `tanh`, `minimum`, `maximum`, `clip`, `where`;
- reductions and layouts: `sum`, `mean`, `min`, `max`, `reshape`, `transpose`,
  `stack`, `concatenate`, `einsum`;
- physical tensor operations: `mag`, `symm`, `dev`, `dot`, `ddot`.

For compatibility, existing declaration methods under `fno.Math` remain
available. New code should use `fno.Field.grad("U")` for a native
`FieldExpression`, while `fno.Math.symm(array)` evaluates immediately using the
array's backend. Spatial derivatives of arbitrary arrays are not approximated
in Python because they require the OpenFOAM mesh and discretisation scheme.

Expressions compose without manually constructing strings:

```python
inputs = {
    "velocity_gradient": fno.Field.grad("U"),
    "convection": fno.Field.div("phi", "U"),
    "thermal_diffusion": fno.Field.laplacian("alphaEff", "T"),
    "vorticity": fno.Field.curl("U"),
    "strain": fno.Math.dev(fno.Math.symm(fno.Field.grad("U"))),
    "gradient_alignment": fno.Math.dot(
        fno.Field.grad("T"),
        fno.Field.grad("c"),
    ),
}
```

The finite-volume operators use the copied case's `system/fvSchemes`. During
case preparation FoamNordic preserves every exact scheme supplied by the user.
If an exact entry is absent, a usable section `default` is preserved. Only when
both are unavailable or the default is `none` does FoamNordic add an exact
fallback from `derivedSchemes.json` inside the isolated run case. The source
case is never changed. Current fallbacks are `Gauss linear` for gradient,
curl, and divergence expressions and `Gauss linear corrected` for Laplacians.

This matters particularly for conservative expressions such as
`div(phi, U)`: an existing solver scheme wins over the fallback. Unary
`div(U)` is also supported, but requires either `div(U)` or a non-`none`
`divSchemes/default`; FoamNordic supplies the exact fallback when necessary.

This separation lets the same numerical function run with NumPy for a small
test and with JAX for tracing or export without silently replacing OpenFOAM's
finite-volume operators.
