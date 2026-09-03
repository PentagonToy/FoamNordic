# Native runtime

FoamNordic treats every resident model as a field-program contract, not as a
particular ML framework or physical role. Each exchange declares named input
and output fields, element types, component counts, exchange index, solver time
index, physical time, and local cell count.

The native state machine is:

```text
waiting
  → collecting_inputs
  → inputs_ready
  → evaluating
  → outputs_ready
  → completed
```

Missing, duplicate, stale, out-of-order, incorrectly shaped, or incorrectly
typed fields stop the exchange before inference. Exchange indices must increase
monotonically. A returned tensor must match both the active exchange index and
its physical time; a complete but stale response is rejected before any
solver-owned output field is modified.

## Closure examples

Progress-variable combustion can use:

```text
inputs:  c, c_var, T
output:  omega_c
```

An algebraic LES closure can use:

```text
inputs:  grad_U (9 components), delta
output:  nut
```

A k-equation closure can add resolved strain, filter width, density, or the
modeled energy field without changing the exchange engine.

## Native bypass

Bypass is a first-class native policy with three responsibilities:

1. select the cell indices that require model evaluation;
2. prefill outputs for bypassed cells;
3. scatter model predictions from active-cell order back to mesh-cell order.

For combustion, a policy may bypass cells outside the progress-variable range,
with negligible variance, below an ignition temperature, or covered by an
analytic or beta-FDF rule. For LES, a policy may apply a laminar, near-wall,
realizability, or clipping rule. These physics policies are separate C++ or
Fortran kernels; the state machine only validates their indices and complete
outputs.

The model therefore receives only active cells. Python never builds the mask,
copies the full mesh into a filtered temporary array, or scatters the result in
a timestep loop.

## Native runner

`InferenceRunner` owns the complete hot path for one or more exchanges:

```text
Harbor input tensors
  → validate one ordered exchange batch
  → seal required program inputs
  → run the native bypass policy
  → evaluate active cells through a ModelKernel
  → merge predictions into full mesh fields
  → Harbor output tensors + completion marker
```

The runner treats the Rune `complete` message as the batch boundary. Every
input tensor in that batch must share its exchange index, solver time index,
physical time, and
cell count. A `shutdown` message is accepted only between exchanges; shutdown
during a partial batch is an error rather than an implicit incomplete result.

`ModelKernel` is deliberately smaller than a framework API. It receives the
validated input map, the sorted active-cell indices, exchange index, and
physical time, and returns named prediction tensors. Native ONNX, a compiled
model, or a managed Python worker may implement that boundary without changing
OpenFOAM coupling, bypass rules, or Rune/Fjord transport.

An input and output may intentionally share a field name, such as `U -> U`.
This represents atomic replacement of an OpenFOAM field after native inference;
names remain unique within the input list and within the output list.

## Resident worker

`ModelWorker` owns one native listener and one complete model session.
It validates the model manifest before accepting a solver, negotiates UDS or
TCP, upgrades a same-node UDS session to SHM when both peers support it, and
then gives the established Harbor to `InferenceRunner`. Shutdown is a Rune
lifecycle message processed between atomic exchanges. The worker therefore
contains no timestep polling and leaves no Unix socket after its lifetime.
If validation or inference fails during an active exchange, the worker sends a
Rune error boundary before terminating. The solver rejects that exchange
immediately and leaves every registered output field unchanged; it does not
wait for a missing completion marker or apply a partial prediction.
This error boundary is exercised over both the socket-backed channel and the
native SHM ring used after a same-node upgrade.

The current end-to-end native test sends `c`, `c_var`, and `T`
through a real Fjord socket channel. It evaluates only cells selected by
`c_var`, reconstructs the full `omega_c` field, and verifies the matching
completion marker.

Solver-integrated closures always use per-call sequencing. The transport
handshake establishes one persistent blocking session, while every invocation
of `correctNut()`, a modeled-equation closure update, or a combustion source
evaluation receives a fresh monotonic exchange index. Multiple calls at the
same OpenFOAM time index and physical time are intentional and must each wait
for their committed response. Timestep deduplication belongs only to the
general function-object field path.

`FieldProgramPort` is the solver-facing boundary for this rule. A solver begins
an invocation with its OpenFOAM time index and physical time, provides any number
of read-only field views, registers mutable output views, and commits once.
Commit performs the complete blocking atomic exchange; it returns only after
every declared output has been validated and copied into the solver-owned
memory. The port assigns its own monotonic exchange index, so neither PIMPLE
outer-corrector calls nor repeated combustion-source evaluations are collapsed
merely because their OpenFOAM time metadata is identical.

The port depends only on named tensor views and an `ExchangeContract`. It has
no knowledge of `nut`, modeled kinetic energy, progress variables, reaction
rates, turbulence classes, or combustion classes. OpenFOAM integrations should
therefore remain thin view adapters; adding a new closure changes its contract
and physics kernel rather than the transport state machine.

## Language boundary

C++ owns field views, exchange state, memory, and communication. Managed
Python workers receive validated dense arrays only at the model boundary;
they do not own OpenFOAM fields or lifecycle state.

Native diagnostic lines use this form:

```text
[FoamNordic] Info: Model exchange ready.
```

The tag is yellow only when stderr is attached to a terminal. Batch logs and
redirected files contain no ANSI escape sequences.
