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
FoamNordic manifest inside the `.fnom` bundle. Its versioned header is followed by the model family,
artifact path, ordered input/output contracts, affine scaler coefficients, and
optional Equinox leaf table. The C++ loader places explicit limits on file,
string, field, feature, rank, and leaf counts; it rejects truncation, invalid
enum values, malformed scaler flags, and trailing bytes before a worker starts.
The original model bytes follow that manifest in the same file. New bundles
use the `FNOBND2` container header and begin the payload at a 64-byte boundary;
the reader remains compatible with legacy `FNOBND1` bundles.

`.fnom` is deliberately not compressed. The bounded manifest remains at most
64 MiB, while the embedded payload may be much larger. Native C++ validates
the bundle lengths and manifest before a worker starts. Avoiding an archive
layer removes decompression, temporary extraction, and another runtime
dependency. FNOM v2 adds an optional Joblib resident-runtime field while the
reader remains compatible with split v1/v2 artifacts written by earlier
FoamNordic versions. New exports contain exactly one deployable file.

Before evaluation, C++ gathers only the active cell indices selected by the
bypass policy, packs fields in manifest order into one contiguous feature
matrix, and applies the input scaler in place. It inverse-scales the packed
prediction matrix and splits it back into named closure fields. Consequently,
all three artifact formats share exactly the same preprocessing hot path.

## Native scaler representation

Fitted scikit-learn `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`,
`RobustScaler`, and affine `FunctionTransformer` instances are normalized once
by the nanobind export factory:

```text
transformed = value * gain + bias
original    = (transformed - bias) / gain
```

The mapping from fitted scikit-learn attributes is:

| Scaler | Native gain | Native bias |
|---|---:|---:|
| StandardScaler | `1 / scale_` | `-mean_ / scale_` |
| MinMaxScaler | `scale_` | `min_` |
| MaxAbsScaler | `1 / scale_` | `0` |
| RobustScaler | `1 / scale_` | `-center_ / scale_` |
| FunctionTransformer | inferred feature-wise multiplier | inferred translation |

MinMax clipping is represented explicitly and applied only in the forward
transform. The C++ `Scaler` interface owns `transform` and
`inverse_transform`; its serialized `AffineScaler` implementation performs
input scaling and output inverse scaling on float32 or float64 buffers. The
same code is used for Equinox, Joblib, and ONNX.

`FunctionTransformer` is accepted only when numeric probes demonstrate an
invertible, shape-preserving, feature-wise affine mapping. Nonlinear and
feature-mixing functions fail during export instead of silently changing
runtime semantics.

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

The embedded Equinox payload contains a cloudpickled reconstructable PyTree
and a `batched` call policy. It is loaded once and wrapped with
`jax.jit`; the default policy applies `jax.vmap` over active cells. The
ClosureHost CPU budget configures the XLA CPU runtime before JAX is imported.
This is
less portable than ONNX and must only be used with trusted payloads and a
compatible Python/JAX/Equinox environment.

## Joblib boundary

Joblib remains a Python artifact format; C++ does not attempt to reproduce
pickle or scikit-learn internals. A managed model worker loads it once. Native
FoamNordic still owns field validation, active-cell selection, feature packing,
scaling, request ordering, output inverse scaling, bypass merge, and shutdown.

This isolates Python overhead to model evaluation and prevents the earlier
design from moving full OpenFOAM fields through Python for every operation.
The embedded `.joblib` stream is uncompressed and loaded once when the managed
worker starts. On Linux and macOS, an aligned `FNOBND2` payload is opened
through the process descriptor namespace and its arrays use `mmap_mode="r"`
directly against the single `.fnom` file. FoamNordic verifies that reopening
the descriptor preserves the payload offset before enabling this path. Kernels,
filesystems, or legacy bundles that do not provide that behavior automatically
fall back to streaming the payload into private temporary storage and mapping
that file. Both paths avoid retaining a complete embedded byte stream plus a
decoded copy. Staging, when required, is a one-time startup operation outside
the exchange loop and follows the cluster's normal `TMPDIR` policy.
Joblib payloads are pickle-based and must be trusted. Legacy split artifacts
retain their direct `mmap_mode="r"` startup path.

At startup the managed worker applies its Longship CPU budget to fitted
estimators exposing `n_jobs`, including fitted children of voting and pipeline
models, and bounds common OpenMP, BLAS, and NumExpr thread pools. Estimators
without a parallel prediction implementation remain single-threaded by their
own design; allocating additional CPUs cannot make such a kernel parallel.

`pip install foamnordic` installs the Joblib/scikit-learn and JAX/Equinox
runtimes together. Backend selection is a model-artifact decision rather than
an installation profile.

Joblib export may select the resident execution runtime without changing
`Operator.model()` or the solver declaration:

```python
artifact = fno.Export.joblib(
    model,
    path="smagorinsky.fnom",
    inputs={"features": fno.Tensor.vector(components=10)},
    outputs={"nut": fno.Tensor.scalar()},
    runtime="sklearnex",
)
```

The choice is stored in FNOM v2. The worker activates `patch_sklearn()` before
loading the Joblib payload; the default `runtime="sklearn"` performs no patch.
The Intel extension is installed automatically on its supported Linux x86-64
platform. An artifact that explicitly requests `sklearnex` fails clearly on an
unsupported worker instead of silently changing its runtime. This setting can
accelerate supported estimator kernels, but it does not alter model size,
statistical validity, or the asymptotic lookup cost of a K-nearest-neighbor
model.

## ONNX boundary

ONNX is the preferred fully native deployment format. The ONNX Runtime adapter
implements `ModelKernel`, while the same native scaler and closure runner
remain outside the graph. Keeping preprocessing in FoamNordic avoids generating
different ONNX graphs for every supported scikit-learn scaler.

FoamNordic targets ONNX Runtime **1.28.0** and C API level 28 exactly for the
first native implementation. `foamnordic build` verifies and caches the
official archive for Linux x86_64, Linux aarch64, or Apple Silicon, then
installs the required library beside the ABI-specific ClosureHost. Offline
systems can select an unpacked distribution with
`FOAMNORDIC_ONNX_RUNTIME_ROOT`; `--without-onnx` explicitly omits this path.
The adapter uses only the stable session and
tensor API, not the experimental Model Package API. It requires one packed
rank-two input and one packed rank-two output and accepts float32 or float64.
The session uses Longship's per-program CPU budget as its intra-operator thread
count, keeps the inter-operator pool at one, and retains sequential graph
execution. The default budget is one CPU. ONNX Runtime is not embedded in
the Python wheel because the native host must match the selected platform and
OpenFOAM runtime.

`load_model(manifest_path)` passes an embedded ONNX byte buffer directly to
ONNX Runtime and owns both that buffer, the packed ONNX kernel, and the
field-aware wrapper. Legacy manifests still resolve their relative payload
path beside the manifest.
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
`--connections N` lets one node-local worker serve N solver ranks while owning
one model instance. `--ready-file PATH` publishes a lifecycle marker only after
the listener and model are initialized; `{rank}` in that path expands from the
active Slurm or MPI task identity.
