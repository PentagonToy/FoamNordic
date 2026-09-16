# Observations and retention

## Monitoring and field execution

FoamNordic separates three operations that may appear together in an interactive workflow:

| Operation | Data | May affect solver state |
| --- | --- | :---: |
| Field programme | Required input and output tensors | Yes, through an adapter-owned atomic commit |
| Observation | Bounded summaries and timing | No |
| Artefact retention | Explicit stored fields, checkpoints and visualisation files | No |

Field programmes run in the node-local execution plan. A notebook consumes read-only summaries and never advances the solver.

The interface remains iterative without owning solver progress:

```python
observations = fno.Observe(
    summaries={"U": ("min", "max"), "p": ("min", "max")},
    interval=100,
)

for observation in run.observe(progress=True):
    table.update(observation.summary)
```

Stopping the loop, restarting a kernel, or rendering slowly does not pause a field-programme invocation.

## Retention domains

| Domain | Storage | Backpressure | Intended content |
| --- | --- | --- | --- |
| Exchange | Bounded Rune or shared-memory slots | Required and atomic | Current model request and response |
| Observation | Bounded summary ring | Lossy; never blocks field execution | Reductions and timing |
| Artefact | Filesystem | Controlled by solver write and user policy | Full fields, checkpoints and figures |

### Exchange retention

Rune and shared memory retain only the bounded slots required for an active atomic exchange.

### Observation retention

Compact summaries use a bounded ring. `Observe` exposes only the solver-friendly `interval`; a slow consumer cannot delay a Closure or Transform.

Useful policies are:

```text
interval=1                         observe each exchange
interval=100                       sparse monitoring
no Observe declaration             no live observation
```

### Artefact retention

Full fields, checkpoints, VTK output, and figures are explicit filesystem artefacts fetched on demand.

## Node-local reduction

Before sending an observation across nodes, the node-local adapter reduces the requested field to the public summary contract.

| Reduction | Meaning |
| --- | --- |
| `min` | Minimum rank-local value, followed by the global minimum |
| `max` | Maximum rank-local value, followed by the global maximum |
| `mean` | Count-weighted global arithmetic mean |
| `l2` | Global Euclidean norm |

MPI-wide summaries combine compact partial reductions and never gather complete rank arrays into the orchestrator. Spatial fields, patches, planes and surfaces remain explicit solver or postprocessing artefacts rather than observation records.

## Observation backpressure

Observation transport is separate from field-programme transport and is lossy but bounded:

```text
producer faster than viewer
  → keep newest permitted records
  → increment dropped-observation counter
  → continue solver and field-programme execution
```

`ExecutionPlan` owns field writers and `ObservationPlan` accepts read-only views. Admission is non-waiting: contention, an oversized record, or a full buffer drops the record rather than delaying the solver. `Longship` merges node-local streams in source order; consumers combine records by exchange metadata when they require a timestep-wide summary.

## Bypass is a different policy

Physics bypass selects cells that do not require model evaluation and supplies their valid output analytically. Observation sampling selects data that a user wants to inspect. They may reuse native masks and reductions, but their semantics and failure behaviour are separate: dropping a plot must never change the modelled field.

## Notebook memory target

A run-control notebook fits within a 4 GiB allocation because it owns:

- immutable plan metadata;
- compact job and timing state;
- bounded summaries;
- at most the declared reduced snapshots;
- no OpenFOAM full-field history and no resident model parameters.

JAX training arrays, optimiser state, synthetic datasets, and model export belong to a separate model-build notebook or batch task. A single document may link the resulting artefact into a Longship order, but training is not part of the run-control process.
