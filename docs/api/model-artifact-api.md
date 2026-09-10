# FNOM artefact API

`fno.Export.sklearn()`, `fno.Export.equinox()`, `fno.Export.onnx()`, and `fno.Export.table()` create self-contained `.fnom` files. `fno.Operator.model()` declares an executable model and `fno.Operator.table()` declares a data-only lookup table; inspection returns metadata rather than raw backend bytes.

```python
import foamnordic as fno

artifact = fno.Models.load("reaction-rate.fnom")

print(artifact.name)
print(artifact.backend)
print(artifact.inputs)
print(artifact.outputs)
print(artifact.trust_policy)

artifact.validate()
```

`FnomArtifact` is immutable. `validate()` re-reads the file through the native FNOM parser without importing or executing its embedded backend payload. Joblib, Equinox, and portable C++ payloads are classified as code-bearing; ONNX, native forest, and regular-grid table payloads are classified as data-only. `foamnordic inspect` reports this distinction. Calling `Operator.model()` or `foamnordic compile` is the execution decision; no redundant trust flag is required.

OpenFOAM scalar precision and model precision may differ. The native exchange boundary accepts `float32` and `float64` model contracts with either single- or double-precision OpenFOAM, and performs the required conversion while packing inputs and committing outputs. Equal-precision fields keep the direct copy path; a precision conversion is necessarily not zero-copy.

The equivalent CLI commands are:

```console
foamnordic inspect reaction-rate.fnom
foamnordic inspect reaction-rate.fnom --json
foamnordic validate reaction-rate.fnom
foamnordic diff reference.fnom candidate.fnom
foamnordic diff reference.fnom candidate.fnom --json
```

`diff` validates both artefacts, compares their complete execution manifests and streams embedded or external payloads through SHA-256. File names, external payload paths, and container padding do not affect equivalence, and neither model payload is executed.

See the [FNOM format specification](../architecture/fnom.md) for binary layout and trust boundaries.

## Compiled estimator backend

`sklearn()` selects the compiled backend for supported fitted estimators and uses Joblib for other sklearn-compatible models:

```python
artifact = fno.Export.sklearn(
    model,
    path="smagorinsky.fnom",
    inputs={"features": fno.Tensor.vector(components=10)},
    outputs={"nut": fno.Tensor.scalar()},
    x_scaler=scaler_X,
)
```

The first load asks BindNordic to compile portable C++ into the target-specific cache; later loads reuse it. Version 1 supports scalar-output linear and forest models, KNN, scikit-learn gradient boosting, XGBoost, LightGBM, and voting graphs composed from supported estimators. XGBoost and LightGBM are needed only while exporting their fitted models; the resulting FNOM has no dependency on either runtime. Selection can be fixed for reproducibility with `backend="compiled"` or `backend="joblib"`; the default is `backend="auto"`. The `runtime` argument applies only to Joblib selection.

Warm the cache before entering an HPC allocation when the login and compute nodes share the same target ABI:

```console
foamnordic compile smagorinsky.fnom
```

## Regular-grid tables

`table()` stores ordered scalar coordinate axes and one dense output array. The last value dimension is the packed width of the declared outputs. BindNordic performs multilinear interpolation natively; no Python process or domain library is present in the solver loop.

```python
artifact = fno.Export.table(
    {"progress": c_axis, "variance": g_axis},
    manifold,
    path="manifold.fnom",
    outputs={"state": fno.Tensor.vector(components=8)},
    bounds={"progress": "clamp", "variance": "error"},
)

operator = fno.Operator.table(artifact)
```

Axis order is the input contract. Output names and widths are the output contract. The table is domain-neutral: thermochemical manifolds, material properties, control maps, and other regular-grid data use the same artefact and kernel. Each axis must be finite and strictly increasing. Bounds may be `clamp` or `error`, globally or per axis.
