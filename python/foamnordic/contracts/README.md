# FoamNordic contracts

This directory contains declarative contracts that must ship with the Python
package. Keep solver- or adapter-specific field names and value kinds here,
not in orchestration code.

`openfoam_adapters.yaml` describes built-in OpenFOAM equation adapters. A field
already present in the source case is discovered from its OpenFOAM header; the
YAML contract supplies metadata for solver-owned fields that do not exist until
the adapter constructs them. Adapter names and field names are case-sensitive.

When adding an adapter:

1. declare every native input and output with its physical `kind`;
2. keep the YAML names consistent with the C++ adapter and case template;
3. add a contract test and a solver-integrated acceptance test.

Transport widths are derived from `FieldLayout`; do not store numeric component
counts in this file.
