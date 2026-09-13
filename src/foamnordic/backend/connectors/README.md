# Model connectors

A native connector implements `ModelConnector`: a stable identifier, a supported FNOM model format, and a loader that returns the common BindNordic packed-tensor kernel. It adapts that kernel to ModelHost only; OpenFOAM fields, exchange ordering, placement, and observations remain in FoamNordic, while model scaling and execution remain in BindNordic.

`onnx/` is the native connector. BindNordic loads compiled C++ estimators directly. Joblib and Equinox execute through BindNordic inside the managed Python resident, so they do not need empty C++ connector directories. All backends consume one self-contained FNOM artefact and the same tensor contract.

The base connector ABI remains host-only and synchronous. A connector may additionally export `foamnordic_connector_memory_v1` to declare input and output memory spaces, persistent device state, completion mode, shape-sensitive workspace, preferred batch rows and alignment. Absence of this symbol preserves the host-only defaults.

## Extension rule

A native backend adds a connector and registers it here without modifying Fjord, Longship or an OpenFOAM adapter. Connector-specific model semantics must remain in BindNordic.
