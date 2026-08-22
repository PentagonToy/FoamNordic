# Model artifacts and native preprocessing

FoamNordic supports three user-facing model artifact families:

- **Equinox** for JAX/Equinox parameter trees;
- **Joblib** for scikit-learn estimators and compatible Python models;
- **ONNX** for portable native inference.

These are storage and loader formats, not three different field-exchange
systems. At save time each format is converted into one FoamNordic bundle with
the same closure contract, ordered features, dtypes, shapes, scaler metadata,
and model-kernel boundary. The timestep loop never inspects a Python estimator
or scikit-learn scaler.

The framework-neutral metadata is encoded as a deterministic little-endian
FoamNordic manifest. Its versioned header is followed by the model family,
artifact path, ordered input/output contracts, affine scaler coefficients, and
optional Equinox leaf table. The C++ loader places explicit limits on file,
string, field, feature, rank, and leaf counts; it rejects truncation, invalid
enum values, malformed scaler flags, and trailing bytes before a worker starts.
Model payloads remain separate from this small control artifact.

Before evaluation, C++ gathers only the active cell indices selected by the
bypass policy, packs fields in manifest order into one contiguous feature
matrix, and applies the input scaler in place. It inverse-scales the packed
prediction matrix and splits it back into named closure fields. Consequently,
all three artifact formats share exactly the same preprocessing hot path.

## Native scaler representation

StandardScaler, MinMaxScaler, and RobustScaler are normalized once to:

```text
transformed = value * gain + bias
original    = (transformed - bias) / gain
```

The mapping from fitted scikit-learn attributes is:

| Scaler | Native gain | Native bias |
|---|---:|---:|
| StandardScaler | `1 / scale_` | `-mean_ / scale_` |
| MinMaxScaler | `scale_` | `min_` |
| RobustScaler | `1 / scale_` | `-center_ / scale_` |

MinMax clipping is represented explicitly and applied only in the forward
transform. C++ performs input scaling and output inverse scaling on float32 or
float64 buffers. The same code is used for Equinox, Joblib, and ONNX.

The bundle rejects zero or non-finite gains, mismatched feature counts, integer
input tensors, and incomplete clipping ranges before a simulation starts.

## Equinox trees

An Equinox artifact contains a deterministic flattened leaf manifest. Every
leaf records its stable path, dtype, shape, byte offset, and byte count. The
native loader validates unique paths, non-overlapping ordered storage, and
exact shape-to-byte agreement.

Python is responsible only for flattening and reconstructing the Equinox tree
at bundle creation or managed worker startup. Python pickles and arbitrary
PyTree traversal never occur in the OpenFOAM exchange loop. A future compiled
JAX executable can consume the same manifest without changing the bundle.

## Joblib boundary

Joblib remains a Python artifact format; C++ does not attempt to reproduce
pickle or scikit-learn internals. A managed model worker loads it once. Native
FoamNordic still owns field validation, active-cell selection, feature packing,
scaling, request ordering, output inverse scaling, bypass merge, and shutdown.

This isolates Python overhead to model evaluation and prevents the earlier
design from moving full OpenFOAM fields through Python for every operation.

## ONNX boundary

ONNX is the preferred fully native deployment format. An ONNX Runtime adapter
will implement `ModelKernel`, while the same native scaler and closure runner
remain outside the graph. Keeping preprocessing in FoamNordic avoids generating
different ONNX graphs for every supported scikit-learn scaler.

FoamNordic targets ONNX Runtime **1.28.0** and C API level 28 exactly for the
first native implementation. The adapter uses only the stable session and
tensor API, not the experimental Model Package API. It requires one packed
rank-two input and one packed rank-two output, accepts float32 or float64, and
defaults both ORT thread pools to one thread with sequential execution to avoid
oversubscribing OpenFOAM or Slurm allocations. ONNX support is optional and is
enabled only with `FOAMNORDIC_ONNX_RUNTIME=ON` and an external
`FOAMNORDIC_ONNX_RUNTIME_ROOT`; ONNX Runtime is not vendored.

`load_model(manifest_path)` resolves a relative artifact path beside the
manifest and owns both the packed ONNX kernel and the field-aware wrapper.
`NativeClosureWorker(address, manifest_path, bypass)` uses this loader
directly, so the resident process needs no Python model object: its complete
hot path is field packing, optional input scaling, ONNX inference, optional
output inverse scaling, field unpacking, and atomic publication in native C++.

With `FOAMNORDIC_RESIDENT_TOOLS=ON`, the same path is exposed as a standalone
process:

```text
foamnordic_closure_worker unix:///run/user/1000/closure.sock model.fnom
```

Unix endpoints negotiate SHM automatically and retain UDS as their control
plane. `--no-shm` is available for diagnosis. TCP endpoints remain available
for separated placement; they never advertise same-node SHM.
