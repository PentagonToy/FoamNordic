# Combustion API

The combustion namespace captures scientific ownership independently of a
particular solver family. It validates and serializes declarations. The native
`reactionRateFjord` adapter now supplies the first equation-level boundary: it
evaluates one learned reaction-rate source when an OpenFOAM combustion model's
`correct()` method is called.

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

## Native reaction-rate adapter

`reactionRateFjord` is registered for both `psiReactionThermo` and
`rhoReactionThermo`. Its dictionary binds the semantic inputs `progress`,
`variance`, and `temperature` to case expressions and declares exactly one
solver-owned scalar output. The adapter validates the output field class and
dimensions before opening the Fjord session.

The output dimensions are intentionally explicit. A normalized progress rate
may use `[0 0 -1 0 0 0 0]`, while a volumetric mass source may use
`[1 -3 -1 0 0 0 0]`; the solver and model artifact must agree. The concrete
dictionary template is
`src/foamnordic/template/openfoam/combustion-model/reactionRateFjordProperties.in`.

This is deliberately not a complete combustion solver. `R(Y)` and `Qdot()`
return zero, the adapter does not call `thermo.correct()`, and it does not yet
evaluate the beta-FDF manifold. A progress-variable solver must own the
transport equations, consume the produced reaction-rate field, update the
manifold, and correct thermodynamics in the ordering defined by the native
contract. Consequently, selecting this model in stock `reactingFoam` does not
implement progress-variable combustion.
