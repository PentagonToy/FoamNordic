# Declarative execution plans

## Longship writes the order; it does not cook the data

Longship owns immutable configuration and lifecycle. It does not evaluate
fields, scale arrays, call a model, or participate in a closure timestep. Its
output before launch is a validated order containing:

- allocation and node placement;
- solver and ClosureHost commands;
- model artifact identities;
- ordered OpenFOAM input expressions and output fields;
- native scaling, bypass, clipping, and transformation policies;
- observation cadence and retention limits;
- transport preferences and failure policy.

This order is compiled and distributed before OpenFOAM starts. Each node-local
ClosureHost then owns the executable plan for the lifetime of the solver job.

## User code remains outside the solver loop

The current public declaration family is declarative. The following is
architecture pseudocode, deliberately independent of exact public method
names:

```python
closure = model.bind(
    inputs={"velocity_grad": "grad(U)", "filter_width": "delta"},
    outputs={"eddy_viscosity": "nut"},
)

velocity_adjustment = operation.scale(
    field="U",
    factor=1.00005,
)

longship = experiment.plan(
    case=case,
    closures=[closure],
    operations=[velocity_adjustment],
    placement="attached",
)

longship.run()
```

The public API expresses this through `Longship`, `Closure`, `Transform`, and
`Observe`. The architectural property is invariant: there is no Python `for
step` loop in the production closure path. A transform is compiled to a
node-local native operation, and a closure becomes a native field and executor
contract.

## Compile phase

Before submission, the plan compiler:

1. resolves every OpenFOAM expression and field type;
2. verifies model input/output names, components, dtype, and scalers;
3. orders dependent operations as a directed acyclic graph;
4. rejects duplicate writers and undeclared mutable outputs;
5. selects attached or central execution explicitly;
6. resolves transport capabilities without opening a live exchange;
7. writes rank-to-host and artifact maps;
8. produces a stable plan digest for logs and restart checks.

An invalid plan fails before scheduler resources are consumed whenever local
metadata is sufficient to decide it.

## Runtime phase

For every solver invocation, native code performs:

```text
OpenFOAM call
  → evaluate declared field expressions
  → expose or pack native views
  → execute bypass / transform / model nodes
  → validate all declared outputs
  → publish one atomic completion
  → apply outputs to solver-owned fields
```

The plan is fixed during an active invocation. A control-plane update may
create a new version for a later safe boundary, but cannot mutate an exchange
that has begun.

## Python executor is not a Python callback

Equinox and Joblib fallback may require a persistent Python process beside the
ClosureHost. That process loads one declared artifact and implements the stable
executor interface. It is not arbitrary notebook code called through an
orchestrator on every timestep.

Arbitrary callbacks are diagnostic mode only. They require an explicit remote
callback declaration, report their network and Python hot-path cost, and are
never selected automatically.

## Observation is not computation

A notebook may iterate over sparse summaries or snapshots after the user asks
for observation. Those observations do not own closure progress and cannot
modify solver fields implicitly. A requested modification must be part of the
precompiled operation plan or a versioned plan update at a safe boundary.

Observation sampling and bounded retention are specified in
[Observations and retention](observations-and-retention.md).
