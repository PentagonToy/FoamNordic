# Equinox connector

Equinox uses the JAX resident execution boundary plus deterministic tree-leaf
metadata from FNOM. Tree reconstruction happens once at worker startup, never
inside an OpenFOAM exchange.
