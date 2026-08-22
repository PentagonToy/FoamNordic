# Equinox connector

Equinox uses the JAX resident execution boundary plus deterministic tree-leaf
metadata from FNOM. The reconstructable PyTree is loaded once, wrapped with
`jax.jit`, and reused. Models are vectorised over active cells by default;
already-batched models can opt out during export. Tree reconstruction and JAX
compilation never occur inside an OpenFOAM exchange.

The payload is deliberately path-backed and uncompressed. Equinox artifacts
contain Python reconstruction data and must only be loaded from trusted
sources. They are version-sensitive deployment artifacts; ONNX remains the
portable choice when conversion is practical.
