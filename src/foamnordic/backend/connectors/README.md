# Backend connectors

A native connector implements `ModelConnector`: a stable identifier, a supported FNOM model format, and a loader that returns the common BindNordic packed-tensor kernel. It adapts that kernel to ModelHost only; OpenFOAM fields, exchange ordering, placement, and observations remain in FoamNordic, while model scaling and execution remain in BindNordic.

`onnx/` is the native connector. BindNordic loads compiled C++ estimators directly. Joblib and Equinox execute through BindNordic inside the managed Python resident, so they do not need empty C++ connector directories. All backends consume one self-contained FNOM artifact and the same tensor contract.

A future native backend adds a connector and registers it here without modifying Fjord, Longship, or the OpenFOAM adapter.
