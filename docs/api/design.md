# Python API design

## Responsibility

Python describes experiments before launch and inspects compact results
after or between selected exchanges. C++ owns solver-call timing, field views,
feature evaluation, scaling, bypass, inference, atomic publication, and error
handling.

Model training and run control are separate workflows. JAX arrays, optimizer
state, and training datasets belong to a model-build notebook or batch job. A
run-control notebook loads only artifact metadata and should remain usable in
a 4 GiB process allocation.

The installed API covers four roles:

| Role | Responsibility |
| --- | --- |
| Case | Locate and validate an OpenFOAM case and its execution command |
| Model | Export a framework model into a versioned native artifact |
| Closure | Bind OpenFOAM expressions and output fields to a model contract |
| Experiment | Select local or Slurm resources, placement, lifecycle, and data path |

The public facade is implemented by the `foamnordic` package. Its primary
declarations are `OpenFOAM`, `Longship`, `Closure`, `Transform`, `Observe`,
`Export`, and `Postprocess`. Architecture documents label illustrative names
explicitly when they are not public API.

The Python process may run interactively, under Papermill, or as the control
process of a submitted Longship job. In every form it remains an orchestrator:
bulk closure fields stay on solver nodes unless the user explicitly selects a
central accelerator or remote diagnostic callback. See the
[execution topologies](../architecture/execution-topologies.md).

## Binding boundary

The binding implementation uses nanobind. It exposes small owning
configuration values and lifecycle handles, while native field storage,
OpenFOAM objects, `Harbor`, and the closure state machine remain C++-owned.
Bindings may release the GIL while waiting for a launched workload or a native
summary, but a Python callback is never part of a solver closure call.

The module deliberately binds only stable facade types. Internal Rune messages,
SHM ring slots, raw OpenFOAM field views, and ONNX Runtime objects are not a
public Python ABI. NumPy views are permitted only for explicit observations;
their lifetime must be tied to an owning native snapshot rather than solver
memory. This keeps nanobind replaceable at the boundary without changing the
native execution protocol.

## Required properties

- declarations are compiled before the solver starts;
- no Python callback is inserted into every OpenFOAM closure call;
- attached CPU inference defaults to same-node UDS with an SHM upgrade;
- central inference may select the verified UCX data plane across nodes;
  explicit transport requests never silently fall back to TCP;
- model inputs may be OpenFOAM fields or native expressions such as
  `grad(U)`, `laplacian(p)`, `div(U)`, and `curl(U)`;
- each closure invocation blocks for one complete atomic response;
- repeated PIMPLE, PISO, turbulence, or combustion calls at one physical time
  remain distinct exchanges;
- observations are explicit and may use a sparse cadence;
- transformations, closures, and bypass policies are declared before launch;
  Longship compiles the order but never calculates field data;
- production execution exposes no Python `for step` closure loop.
- an optional observation iterator is read-only, byte-bounded, and cannot send
  solver fields or gate closure progress.

## Non-goals

The Python layer will not expose a database as a scientific object, move full
fields through a notebook loop by default, silently fall back from an explicit
transport request, or own solver field memory.

## Stabilization gate

The public Python API is packaged with nanobind bindings and API tests. Local
and Slurm OpenFOAM validation use the same native contracts. The native
behavior documented under [C++ internals](../internals/README.md) remains the
authoritative definition of low-level transport and OpenFOAM ABI behavior.
