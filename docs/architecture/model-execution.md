# Model execution architecture

All executors implement one validated closure contract:

```text
named input tensors
  → optional native scaling and bypass
  → model executor
  → optional inverse scaling, clipping, and scatter
  → named output tensors
```

Transport, OpenFOAM expressions, exchange sequencing, and output publication
remain outside the framework-specific executor.

## ONNX

ONNX Runtime is the primary fully native executor. It runs inside ClosureHost,
owns its session for the job lifetime, and receives already packed buffers.
There is no Python object in the solver loop.

## Equinox

An Equinox artifact contains a deterministic tree manifest and parameters. Two
execution paths are possible:

1. export a compatible model to ONNX and use the native executor;
2. attach a persistent JAX/Equinox Python executor to each ClosureHost node.

The Python executor reconstructs and compiles the model once. Rune payloads are
not pickled, and the notebook is not called per timestep. SHM buffers should be
exposed through a bounded native binding, with synchronization and ownership
remaining in C++.

## Joblib

Joblib is also a persistent attached executor for models that cannot be
converted to ONNX. The model is loaded once in a controlled Python process.
Inputs and outputs use the same native buffer contract as ONNX; serialized
Python model objects are never transported during an active exchange.

Scikit-learn estimators that convert faithfully to ONNX should prefer the
native route. Joblib fallback must report that Python is present in the hot
path so performance comparisons remain honest.

## Declared native operations

Simple field modifications such as

```text
U_out = U * 1.00005
```

are operation plans, not remote callbacks. The Python API records the
expression before launch and C++ evaluates it beside the solver. The same plan
can include clipping, affine scaling, component selection, masks, or a native
physics bypass without moving the field through Python.

Arbitrary Python functions remain possible through an attached persistent
executor, but require an explicit opt-in. A remote notebook callback is a
diagnostic mode with a visible latency warning.

## Executor selection

```text
native operation available       native C++/Fortran kernel
model converts faithfully        ONNX Runtime
Equinox requires JAX semantics   attached Equinox executor
Joblib cannot convert            attached Joblib executor
central GPU explicitly selected  node gateway → UCX/TCP → GPU executor
```

The executor choice does not alter OpenFOAM field names, scaler metadata,
bypass semantics, or atomic completion rules.
