# FNOM model artifacts
FNOM is the deployment artifact for mathematical and learned operators. It
binds a payload to named tensor inputs and outputs, numeric precision, optional
scalers, backend requirements, and runtime metadata.

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
| ONNX | native ONNX Runtime worker |
| Compiled estimator | target-native cached library |
| Native forest | checked data-only tree evaluator |
| Joblib | persistent Python resident |
| Equinox | persistent JAX resident |

Backend choice does not change solver bindings or the Operator.model API.

## Inspection and trust

inspect, validate, and diff parse metadata and payload bounds without loading
or executing the model. Inspection labels ONNX and native forests as data-only,
and Joblib, Equinox, and portable source payloads as code-bearing. Use
code-bearing artifacts only from a trusted source.

```console
foamnordic inspect model.fnom
foamnordic validate model.fnom
foamnordic diff reference.fnom candidate.fnom
foamnordic compile model.fnom
```

diff compares the effective execution manifest and streams payload bytes
through SHA-256. Paths and container padding do not affect equivalence.

## Runtime behavior

The artifact is validated before worker startup. Inputs are packed by declared
name and width, scalers execute inside ModelHost, and outputs are checked before
atomic publication. Compiled payloads are cached by content and target.
Internal encoding fields reject corrupt data; they are not public artifact
generations.
