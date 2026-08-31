# FNOM format and model artifacts

**FNOM (FoamNordic Model, pronounced “ef-nom”) is FoamNordic's
self-contained, backend-neutral execution artifact for deploying analytical
and learned model kernels into coupled simulations.** FNOM does not replace
backend model formats. It binds an ONNX, compiled C++, Joblib, Equinox, or future payload to
the tensor contract, preprocessing metadata, runtime requirements, and
compatibility rules required by FoamNordic.

FoamNordic is the canonical FNOM writer, reader, validator, and execution
runtime. The format is open and may be implemented elsewhere; encryption,
compression, and deliberate obscurity are not part of its ownership model.

## Canonical workflow

```text
fno.Export.* -> model.fnom -> fno.Operator.model() -> ClosureHost
```

Metadata can be inspected without importing or executing the backend payload:

```bash
foamnordic inspect model.fnom
foamnordic inspect model.fnom --json
foamnordic validate model.fnom
```

```python
artifact = fno.Models.load("model.fnom")
artifact.backend
artifact.inputs
artifact.outputs
artifact.validate()
```

`validate` checks the container, bounds, manifest version, tensor contract,
scalers, tree metadata, and payload layout. It deliberately does not unpickle
Joblib or Equinox payloads, compile C++ source, or execute an ONNX graph.
Backend loading remains an execution-time validation performed by ClosureHost.

## Execution boundaries

All backends consume named packed tensors with the same native scaling and
atomic publication contract. ONNX executes in the native ClosureHost.
Supported sklearn graphs are lowered to compiled C++; unsupported estimators
use a persistent Joblib resident. Equinox reconstructs and JIT-compiles its
trusted PyTree once in a persistent JAX resident. Serialized Python objects are
never transported during an exchange, and the notebook is never called from
the solver loop.

Simple declared operations remain native operation plans rather than model
callbacks. Backend selection does not change OpenFOAM field bindings, scaler
metadata, placement, or completion semantics.

## Encoding rules

All integers and IEEE-754 float64 values are little-endian. Strings are a
`u32` byte length followed by that many bytes. Counts precede variable-length
sequences. A valid file ends exactly at the declared payload boundary; trailing
bytes are rejected.

Container version and manifest schema are independent. The current writer
emits the `FNOBND2` container. A manifest uses schema 1 unless it needs the
schema-2 backend runtime field.

### `FNOBND2` container header

| Offset | Width | Meaning |
|---:|---:|---|
| 0 | 8 | magic `FNOBND2\0` |
| 8 | 8 | manifest byte count, `u64` |
| 16 | 8 | absolute payload offset, `u64` |
| 24 | 8 | payload byte count, `u64` |

The binary manifest begins at byte 32. The payload begins at the declared
offset, aligned to 64 bytes, and remains uncompressed. Padding between the
manifest and payload has no semantics. The payload occupies the remainder of
the file.

The legacy `FNOBND1\0` container has a 24-byte header containing manifest and
payload byte counts; its payload follows the manifest without alignment. The
reader accepts it, but the writer never emits it.

### Binary manifest

The manifest begins with `FNOMAN1\0`, followed by:

1. manifest schema, `u32`;
2. backend family, `u8` (`1` Equinox, `2` Joblib, `3` ONNX, `4` compiled C++);
3. backend artifact name string;
4. closure-contract name string;
5. input field sequence;
6. output field sequence;
7. optional input affine scaler;
8. optional output affine scaler;
9. Equinox tree-leaf sequence;
10. backend runtime string when schema is 2 or newer.

Each field is encoded as a name string, element-type `u8`, and component count
`u64`. A scaler contains a presence flag; when present it records kind, feature
count, float64 gain and bias arrays, and optional clipping bounds. An Equinox
leaf records its stable path, element type, shape, byte offset, and byte count.

The native reader bounds manifest size to 64 MiB, each string to 1 MiB, and
field, feature, rank, and leaf counts to 65,536. It rejects unknown enum values,
zero components, duplicate field names, scaler-width mismatches, invalid tree
storage, unsupported schemas, truncation, overlap, misalignment, and trailing
data.

## Compatibility policy

| Container | Manifest schema | Current reader | Current writer |
|---|---:|---|---|
| manifest-only `FNOMAN1` | 1–2 | supported for legacy artifacts | no |
| `FNOBND1` | 1–2 | supported | no |
| `FNOBND2` | 1–2 | supported | yes |

Compatibility support is never removed in a patch or minor release. Removing
a documented reader requires a major release, an explicit release note, and a
deterministic upgrade path. A newer writable format must provide a
non-destructive upgrade path before an older reader is retired.

New readers must reject unsupported versions instead of guessing their
meaning. New writers emit only the current container while readers retain the
documented legacy paths.

## Compiled C++ payloads

A compiled v1 FNOM stores portable C++ source and runtime identifier `cpp-v1`,
not a platform-specific shared library. The first worker on a target compiles
the source with a C++17 compiler. FoamNordic caches the resulting library by
the SHA-256 digest of the source, code-generator version, operating system,
architecture, compiler identity, and optimization flags. Subsequent workers
load the cached library directly.

Keeping target binaries outside FNOM preserves one artifact across macOS
arm64, Linux x86-64, and Linux arm64. Compilation is never part of an exchange
hot path. A cache miss does, however, add startup latency and transient compiler
memory; production deployments should warm the cache before a scheduled run.

```console
foamnordic compile model.fnom
```

Compiled v1 lowers scalar-output `Ridge`, `Lasso`, `ElasticNet`,
`LinearRegression`, `ExtraTreesRegressor`, `RandomForestRegressor`,
`KNeighborsRegressor`, `GradientBoostingRegressor`, `XGBRegressor`,
`XGBRFRegressor`, `LGBMRegressor`, and `VotingRegressor` graphs composed from
supported estimators. KNN accepts uniform or distance weights with Euclidean or
Manhattan distance. Boosting exporters accept scalar regression objectives and
numeric tree splits. Unsupported estimators must remain Joblib artifacts rather
than silently changing semantics.

## Performance and trust

The execution payload is never compressed or encrypted. It is decoded once at
worker startup, loaded directly from memory where the backend permits, and
memory-mapped when Joblib provides a compatible interface. There is no JSON
parsing, mandatory extraction, or per-exchange deserialization in the FNOM
hot path.

FNOM is an execution contract, not a security sandbox. Joblib and current
Equinox payloads may contain pickle-compatible objects, while compiled payloads
contain source code passed to the target compiler. All must come from a trusted
source. Inspection and structural validation do not unpickle or compile them.
