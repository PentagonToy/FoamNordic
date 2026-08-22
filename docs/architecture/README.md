# FoamNordic architecture

FoamNordic separates orchestration, model execution, and field transport so a
notebook, a batch script, or a future service can launch the same native
closure path without becoming part of every solver call.

| Document | Contract |
|---|---|
| [Execution topologies](execution-topologies.md) | Attached, orchestrated, and central-accelerator layouts |
| [Control and data planes](control-and-data-planes.md) | What may cross nodes and which transport carries it |
| [Model execution](model-execution.md) | ONNX, Equinox, Joblib, native operations, and Python fallback |
| [Declarative plans](declarative-plans.md) | Loop-free orchestration and native execution plans |
| [Observations and retention](observations-and-retention.md) | Lightweight monitoring without owning solver progress |

The authoritative default is **node-local inference**: one ClosureHost is
attached to every OpenFOAM solver node inside one Longship allocation. This is
the only topology allowed to become implicit. More expensive topologies must
be requested explicitly and report their selected data path.
