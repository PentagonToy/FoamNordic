# Observations and retention

## Preserve the monitoring experience, not the closure loop

Earlier notebooks used one `for step` loop for three different jobs:

1. receive closure inputs and return required outputs;
2. compute monitoring statistics;
3. retain a field for plotting.

Only the last two belong near an interactive notebook. Closure progress and
field replacement move entirely into the node-local native execution plan.
The notebook becomes a read-only observer and never calls `step.send()`.

An optional future interface may still feel iterative:

```python
monitor = longship.observe(
    summaries={"U": ["min", "max"], "p": ["min", "max"]},
    every=100,
    snapshots={"U": plane(z=0.0, resolution=(256, 256))},
    snapshot_every=1000,
    retention=retention.latest(2, maximum="64 MiB"),
)

for observation in monitor:
    table.update(observation.summary)
    if observation.snapshot is not None:
        plot(observation.snapshot)
```

This loop consumes observation events. Stopping it, restarting the kernel, or
rendering slowly does not pause a closure invocation. It has no mutable solver
field handle.

## Three retention domains

Retention is not one global integer.

### Exchange retention

Rune/SHM retains only the bounded slots required for an active atomic
exchange. A slot is recycled as soon as its consumer advances the sequence.
The default does not preserve ten historical full fields.

### Observation retention

Compact summaries and sampled views use a byte-bounded ring. The policy
declares both record and byte limits. When a nonessential observer falls
behind, the default is `drop_oldest`; closure execution never waits for free
observation memory.

Useful policies are:

```text
latest(1)                    current display only
latest(2, maximum=64 MiB)    double-buffered visualization
every(100, keep=10)          bounded monitoring history
none                         no live observation
```

### Artifact retention

Full fields, checkpoints, VTK output, and final figures are explicit files.
They use filesystem retention and are fetched on demand. A full-field snapshot
is not silently copied into notebook memory merely because monitoring is on.

## Node-local reduction

Before sending an observation across nodes, the node-local host may compute:

- min, max, mean, variance, norms, or histograms;
- convergence and timing counters;
- probes and selected patches;
- planes, lines, and bounded point samples;
- fixed-resolution visualization grids;
- explicitly requested sparse cell selections.

MPI-wide summaries combine compact partial reductions. They do not gather all
rank arrays into the orchestrator. A plane or surface is assembled only for
the declared visualization product.

## Observation backpressure

Observation transport is separate from closure transport. Its default behavior
is lossy but bounded:

```text
producer faster than viewer
  → keep newest permitted records
  → increment dropped-observation counter
  → continue solver and closure execution
```

Synchronous observation is diagnostic-only and must be requested explicitly.
It reports that visualization latency can enter solver wall time.

This separation is also enforced by the native type system. An
`ExecutionPlan` owns field writers and cannot contain observers. An
`ObservationPlan` accepts read-only field views and cannot return modified
values to OpenFOAM. Its publisher uses a non-waiting admission attempt; lock
contention, an oversized record, or a full `drop_newest` buffer increments the
dropped-observation count instead of delaying the solver. With the default
`drop_oldest` policy, the newest bounded summary replaces stale display data.
An independent native relay drains this ring with a condition-variable wakeup
and frames records for Longship. There is no periodic polling and no write to
the observation channel from the solver thread.

Longship may merge any number of node-local receivers into one bounded stream.
Every accepted record receives a monotonic `stream_index` at ingestion while
retaining its source name, exchange index, and physical time. The merge always
preserves each source's order. It deliberately does not delay a faster source
to manufacture a physical-time ordering across nodes: observations are lossy,
and a delayed or dropped source must never become a solver barrier. Consumers
that require a timestep-wide summary combine matching source records by their
explicit exchange metadata and report missing contributions.

## Bypass is a different policy

Physics bypass selects cells that do not require model evaluation and supplies
their valid output analytically. Observation sampling selects data that a user
wants to inspect. They may reuse native masks and reductions, but their
semantics and failure behavior are separate: dropping a plot must never change
the modeled field.

## Notebook memory target

A run-control notebook should work within a 4 GiB allocation because it owns:

- immutable plan metadata;
- compact job and timing state;
- bounded summaries;
- at most the declared reduced snapshots;
- no OpenFOAM full-field history and no resident model parameters.

JAX training arrays, optimizer state, synthetic datasets, and model export
belong to a separate model-build notebook or batch task. A single document may
link the resulting artifact into a Longship order, but training is not part of
the run-control process.
