# Provisional combustion API

The combustion namespace captures scientific ownership before a solver-native
adapter is written. It currently validates and serializes declarations; it
does not yet make stock OpenFOAM solvers solve progress-variable combustion.

```python
reaction_rate = fno.Closure(
    name="reactionRate",
    operator=fno.Operator.model(MODEL_DIR / "reaction-rate.fnom"),
    inputs={
        "progress": fno.field("c_tilde"),
        "variance": fno.field("c_var"),
        "temperature": fno.field("T_tilde"),
    },
    outputs={
        "reaction_rate": fno.field("omega_c"),
    },
)

manifold = fno.Combustion.Manifold.beta_fdf(
    table=MODEL_DIR / "flamelet.fnom",
    progress=fno.field("c_tilde"),
    variance=fno.field("c_var"),
    outputs={
        "species": fno.fields("Y_*"),
        "enthalpy": fno.field("h"),
    },
)

combustion = fno.Combustion.ProgressVariable(
    reaction_rate=reaction_rate,
    manifold=manifold,
)
```

`progress`, `variance`, `temperature`, and `reaction_rate` are semantic port
names. `c_tilde`, `c_var`, `T_tilde`, and `omega_c` are case bindings and may
differ between solver families. The reaction-rate closure and manifold must
bind the same progress and variance fields.

`fno.fields("Y_*")` declares a field-family selection. A future case compiler
will expand it against the copied case's OpenFOAM object registry and reject an
empty match before launch.

The first beta-FDF contract accepts a pre-integrated `.fnom` table only.
Runtime Python quadrature is deliberately excluded from the native hot path.
See the [native combustion contract](../native/combustion-contract.md) for
equation ordering, dimensions, parallel identity, and acceptance requirements.
