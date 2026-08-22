# Math API

`fno.Math` is the backend-neutral mathematical vocabulary used by closure and
transform functions. It keeps NumPy arrays in NumPy and JAX arrays or tracers
in JAX; importing FoamNordic does not install or eagerly import either backend.

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

- OpenFOAM declarations: `field`, `filter_width`, `grad`, `div`, `laplacian`,
  `curl`, `mag`, `symm`, `dev`, `dot`, `ddot`;
- element-wise operations: `abs`, `sqrt`, `square`, `exp`, `expm1`, `log`,
  `log1p`, `sin`, `cos`, `tan`, `tanh`, `minimum`, `maximum`, `clip`, `where`;
- reductions and layouts: `sum`, `mean`, `min`, `max`, `reshape`, `transpose`,
  `stack`, `concatenate`, `einsum`;
- physical tensor operations: `mag`, `symm`, `dev`, `dot`, `ddot`.

Field declarations and array mathematics deliberately share the namespace but
not their representation. For example, `fno.Math.grad("U")` creates a native
`FieldExpression`, while `fno.Math.symm(array)` evaluates immediately using the
array's backend. Spatial derivatives of arbitrary arrays are not approximated
in Python because they require the OpenFOAM mesh and discretisation scheme.

Expressions compose without manually constructing strings:

```python
inputs = {
    "velocity_gradient": fno.Math.grad("U"),
    "convection": fno.Math.div("phi", "U"),
    "thermal_diffusion": fno.Math.laplacian("alphaEff", "T"),
    "vorticity": fno.Math.curl("U"),
    "strain": fno.Math.dev(fno.Math.symm(fno.Math.grad("U"))),
    "gradient_alignment": fno.Math.dot(
        fno.Math.grad("T"),
        fno.Math.grad("c"),
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
