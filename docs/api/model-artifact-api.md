# FNOM artifact API

`fno.Export.sklearn()`, `fno.Export.equinox()`, and `fno.Export.onnx()` create
self-contained `.fnom` files.
`fno.Operator.model()` is the canonical execution entry point; inspection
returns metadata rather than raw backend bytes.

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

See the [FNOM format specification](../architecture/fnom.md) for binary
layout, trust boundaries, and compatibility policy.

## Compiled estimator backend

`sklearn()` selects the compiled backend for supported fitted estimators and
uses Joblib for other sklearn-compatible models:

```python
artifact = fno.Export.sklearn(
    model,
    path="smagorinsky.fnom",
    inputs={"features": fno.Tensor.vector(components=10)},
    outputs={"nut": fno.Tensor.scalar()},
    x_scaler=scaler_X,
)
```

The first load compiles portable C++ into FoamNordic's target-specific cache;
later loads reuse it. Version 1 supports scalar-output linear and forest models,
KNN, scikit-learn gradient boosting, XGBoost, LightGBM, and voting graphs
composed from supported estimators. XGBoost and LightGBM are needed only while
exporting their fitted models; the resulting FNOM has no dependency on either
runtime. Selection can be fixed for reproducibility with `backend="compiled"`
or `backend="joblib"`; the default is `backend="auto"`. The `runtime` argument
applies only to Joblib selection.

Warm the cache before entering an HPC allocation when the login and compute
nodes share the same target ABI:

```console
foamnordic compile smagorinsky.fnom
```
