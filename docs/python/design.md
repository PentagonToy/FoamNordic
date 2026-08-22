# Python API design

## Responsibility

Python will describe experiments before launch and inspect compact results
after or between selected exchanges. C++ owns solver-call timing, field views,
feature evaluation, scaling, bypass, inference, atomic publication, and error
handling.

Model training and run control are separate workflows. JAX arrays, optimizer
state, and training datasets belong to a model-build notebook or batch job. A
run-control notebook loads only artifact metadata and should remain usable in
a 4 GiB process allocation.

The API should eventually cover four roles:

| Role | Responsibility |
| --- | --- |
| Case | Locate and validate an OpenFOAM case and its execution command |
| Model | Export a framework model into a versioned native artifact |
| Closure | Bind OpenFOAM expressions and output fields to a model contract |
| Experiment | Select local or Slurm resources, placement, lifecycle, and data path |

Names and signatures remain provisional until bindings exist.

The Python process may run interactively, under Papermill, or as the control
process of a submitted Longship job. In every form it remains an orchestrator:
bulk closure fields stay on solver nodes unless the user explicitly selects a
central accelerator or remote diagnostic callback. See the
[execution topologies](../architecture/execution-topologies.md).

## Binding boundary

The binding implementation will use nanobind. It will expose small owning
configuration values and lifecycle handles, while native field storage,
OpenFOAM objects, `Harbor`, and the closure state machine remain C++-owned.
Bindings may release the GIL while waiting for a launched workload or a native
summary, but a Python callback must never be part of a solver closure call.

The first module should bind only stable facade types. Internal Rune messages,
SHM ring slots, raw OpenFOAM field views, and ONNX Runtime objects are not a
public Python ABI. NumPy views are permitted only for explicit observations;
their lifetime must be tied to an owning native snapshot rather than solver
memory. This keeps nanobind replaceable at the boundary without changing the
native execution protocol.

## Required properties

- declarations are compiled before the solver starts;
- no Python callback is inserted into every OpenFOAM closure call;
- attached CPU inference defaults to same-node UDS with an SHM upgrade;
- central inference uses TCP until a real UCX channel is implemented and
  verified across two nodes;
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

A public Python API is ready only after bindings, packaging, API tests, local
OpenFOAM examples, and Slurm examples all use the same native contracts. Until
then, native behavior documented under [C++ internals](../native/README.md) is
authoritative.
