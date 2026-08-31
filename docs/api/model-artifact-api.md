# FNOM artifact API

`fno.Export.onnx()`, `fno.Export.joblib()`, and `fno.Export.equinox()` create
self-contained `.fnom` files. `fno.Operator.model()` is the canonical execution
entry point; inspection returns metadata rather than raw backend bytes.

```python
import foamnordic as fno

artifact = fno.Models.load("reaction-rate.fnom")

print(artifact.name)
print(artifact.container_version)
print(artifact.schema_version)
print(artifact.backend)
print(artifact.inputs)
print(artifact.outputs)

artifact.validate()
```

`FnomArtifact` is immutable. `validate()` re-reads the file through the native
FNOM parser without importing or executing its embedded backend payload.

The equivalent CLI commands are:

```console
foamnordic inspect reaction-rate.fnom
foamnordic inspect reaction-rate.fnom --json
foamnordic validate reaction-rate.fnom
```

See the [FNOM format specification](../internals/fnom-format.md) for binary
layout, trust boundaries, and compatibility policy.
