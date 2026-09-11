# FNOM model artifacts
FNOM is the deployment artifact for mathematical and learned operators. BindNordic owns its container, tensor contract, scalers, backend metadata, payload validation, and model execution. FoamNordic exposes the export and operator API and connects the resulting model tensors to solver fields.

```text
trained model ---> fno.Export.* ---> model.fnom
                                          |
                                          v
                               inspect / validate / diff
                                          |
                                          v
                                  Operator.model()
                                          |
                                          v
                                      ModelHost
```

## Backends

| Backend | Execution |
| --- | --- |
| ONNX | native ONNX Runtime connector built by `foamnordic build` |
| Compiled estimator | target-native cached library |
| Native forest | checked data-only tree evaluator |
| Regular-grid table | native multilinear interpolation |
| Joblib | persistent Python resident |
| Equinox | persistent JAX resident |

Backend choice does not change solver bindings or the `Operator.model` API.

### Tree-estimator guidance

Prefer compiled estimators and native forests for small estimators or modest tree and boosting-iteration counts. Large compiled estimators can generate C++ whose compilation cost or memory use outweighs the inference benefit; native forests avoid compilation, but their execution cost still scales with model size. Joblib is the general compatibility choice and avoids generated C++ compilation, although its persistent Python execution can be slower. Benchmark representative workloads before selecting a production backend.

## Inspection and trust

inspect, validate, and diff parse metadata and payload bounds without loading or executing the model. Inspection labels ONNX, native forests, and regular-grid tables as data-only, and Joblib, Equinox, and portable source payloads as code-bearing. Use code-bearing artifacts only from a trusted source.

```console
foamnordic inspect model.fnom
foamnordic validate model.fnom
foamnordic diff reference.fnom candidate.fnom
foamnordic compile model.fnom
```

diff compares the effective execution manifest and streams payload bytes through SHA-256. Paths and container padding do not affect equivalence.

## Runtime behaviour

The artifact is validated before worker startup. FoamNordic packs inputs by declared name and width; BindNordic scales and evaluates model tensors; FoamNordic validates the returned field layout before atomic publication. Compiled payloads are cached by content and target. Internal encoding fields reject corrupt data; they are not public artifact generations.
