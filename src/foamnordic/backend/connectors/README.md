# Backend connectors

A native connector implements `ModelConnector`: a stable identifier, a
supported FNOM model format, and a loader that returns the common packed-tensor
kernel. It loads and evaluates a model only; OpenFOAM fields, scaling, exchange
ordering, placement, and observations remain outside the connector.

`onnx/` is the native connector. Compiled C++ estimators are loaded by the
inference runner directly. Joblib and Equinox execute in the managed Python
resident, so they do not have empty C++ connector directories. All backends
still consume one self-contained FNOM artifact and the same tensor contract.

A future native backend adds a connector and registers it here without
modifying Fjord, Longship, or the OpenFOAM adapter.
