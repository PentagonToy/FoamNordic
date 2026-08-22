# Combustion API

The combustion namespace captures scientific ownership independently of a
particular solver family. It validates the scientific declaration, lowers it
to two resident FNOM programs, and writes one native
`progressVariableFjord` combustion dictionary.

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

longship = fno.Longship(
    case=case,
    combustion=combustion,
    scheduler=scheduler,
)
```

`progress`, `variance`, `temperature`, and `reaction_rate` are semantic port
names. `c_tilde`, `c_var`, `T_tilde`, and `omega_c` are case bindings and may
differ between solver families. The reaction-rate closure and manifold must
bind the same progress and variance fields.

`fno.fields("Y_*")` declares a field-family selection. The case compiler
expands it deterministically against the initial OpenFOAM object registry and
rejects an empty match or a non-scalar selected field before launch. Every
selected field remains solver-owned.

The manifold FNOM manifest uses one scalar output port per expanded family
field (for example `Y_CH4` and `Y_O2`) and retains explicit logical ports such
as `enthalpy`. The family label is declaration-time metadata; expansion
deliberately produces the ordered native tensor contract before workers start.

The first beta-FDF contract accepts a pre-integrated `.fnom` table only.
Runtime Python quadrature is deliberately excluded from the native hot path.
See the [native combustion contract](../native/combustion-contract.md) for
equation ordering, dimensions, parallel identity, and acceptance requirements.

## Native progress-variable coordinator

`progressVariableFjord` and the narrower `reactionRateFjord` are registered for
both `psiReactionThermo` and `rhoReactionThermo`. The progress-variable model
owns two isolated sessions. Each call to `correct()` evaluates the reaction
rate first, evaluates the manifold second, and then performs at most one
`thermo.correct()`. It accepts only the currently implemented `lagged` source
and `outer_corrector` correction policy.

The generated dictionary declares solver-owned scalar outputs. It validates
every output field class, requires exactly one reaction-rate output, prevents
the manifold from overwriting that source, and checks source dimensions before
opening either Fjord session. `Longship` starts both resident workers as one
fail-together host group.

The output dimensions are intentionally explicit. A normalized progress rate
may use `[0 0 -1 0 0 0 0]`, while a volumetric mass source may use
`[1 -3 -1 0 0 0 0]`; the solver and model artifact must agree. The coordinated
dictionary template is
`src/foamnordic/template/openfoam/combustion-model/progressVariableFjordProperties.in`.

This is deliberately not a complete combustion solver. `R(Y)` and `Qdot()`
return zero because a progress-variable solver must own its transported scalar
equations and consume the declared reaction-rate field explicitly. That solver
must call `combustion->correct()` once at its native outer-corrector site after
the progress and variance solves. Selecting this model in stock
`reactingFoam` therefore does not create a progress-variable equation or map
the source into one.
