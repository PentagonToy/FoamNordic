# Native field pipeline

FoamNordic has two deliberately different execution paths.

## Native fast path

The default a-posteriori path keeps an OpenFOAM field in native memory:

```text
OpenFOAM field view
  → fused native feature/field transform
  → native bypass and closure inference
  → native bounds and physical constraints
  → OpenFOAM field update
```

No full field enters Python. Component-wise affine transforms, scaler
application, clipping, native closure kernels, prediction scatter, and
monitoring reductions run in C++ or an attached Fortran kernel. Identity plans
such as `U *= 1.0` are detected and skipped without allocating a destination.

Monitoring sends only compact values such as exchange index, physical time,
minimum, maximum, mean, and timing. A Jupyter table therefore does not require
copying `U`, `p`, `nut`, or `omega_c` every timestep.

## Explicit Python observation

Python access remains available for research and debugging, but it is opt-in:

```text
native view → requested snapshot/view → Python callback or notebook
```

An observation cadence can request every Nth exchange instead of blocking the
solver on every step. A Python callback is never silently inserted into a
native closure plan.

## Field modification

The native adapter supports in-place or destination-based transforms with:

- one affine scale and bias broadcast to every component;
- one scale and bias per scalar/vector/tensor component;
- fused lower and upper bounds;
- float32 and float64 storage;
- native minimum, maximum, mean, and count reductions.

This covers LDC-style field modification, standard or min-max scaling,
component calibration, realizability bounds, and simple source corrections.
More complex operations become named native kernels rather than Python code
executed inside the timestep loop.

## Declarative plan boundary

The Python control plane declares work before launching OpenFOAM. It compiles
`Closure`, `Transform`, `Observe`, and `Longship` declarations into separate
native execution and observation plans. The compact example below is
illustrative pseudocode for that internal plan boundary, not a second public
Python API:

```python
execution = fno.ExecutionPlan()
execution.modify("U", scale=1.005)

observation = fno.ObservationPlan()
observation.observe(["U", "p"], every=500, offset=1)

case.attach(execution, observation=observation)
```

The two plans remain different native types. `ExecutionPlan` is lossless,
blocking where closure correctness requires it, and may mutate a solver field.
`ObservationPlan` is read-only, byte-bounded, and non-blocking by default.
Neither invokes Python from the timestep loop. With `every=500, offset=1`, only
exchanges 1, 501, 1001, and so on publish compact observations. Omitting the
observation plan constructs no observation buffer and sends no monitoring
traffic.

See the [Python API design](../python/design.md) for the implemented ownership
boundary.

## Ownership

`MutableTensorView` is non-owning. OpenFOAM retains the field allocation, and a
native operation may update it directly when the plan permits mutation. A
Python or communication snapshot receives an explicit owner so its lifetime
cannot exceed the native buffer.
