# Backend connector contract

Every model backend implements `ModelConnector`: a stable id, a supported FNOM
model format, and a loader that returns the common packed-tensor kernel. Field
selection, scaling, exchange ordering, MPI placement, and observations remain
outside connectors. Consequently a connector must only load and evaluate its
model; it must not know about OpenFOAM fields or Slurm.

`onnx/` is the fully native reference implementation. `equinox/`, `jax/`, and
`joblib/` document the managed Python-resident backends that will implement the
same packed contract. A future TensorFlow or Torch contributor adds a format to
the next FNOM manifest version, implements one connector, registers it, and
does not modify Fjord, Rune, Harbor, or the OpenFOAM adapter.

Large payloads stay beside the small FNOM manifest and must be opened by path;
connectors must not copy an entire model merely to register it.
