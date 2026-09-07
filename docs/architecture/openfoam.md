# OpenFOAM integration
FoamNordic integrates through a small ABI-specific library built against the active OpenFOAM environment. The wheel supplies the SDK and native runtime; foamnordic build compiles the integration for the selected ABI.

## Field programs

A Transform reads and writes registered fields at a declared solver stage. A Closure enters an equation-level adapter owned by the solver. Both use the same transport, FNOM, resource, and failure contracts.

Supported stored-field layouts include scalar, vector, symmetric tensor, tensor, and explicitly registered surface fields. Derived expressions such as grad, div, curl, laplacian, magnitude, trace, and invariant are evaluated by OpenFOAM from the current mesh and field registry.

## Binding rules

- Logical model port names are separate from OpenFOAM field names.
- Shapes and dimensions are validated before launch and again at exchange.
- Field storage is resolved for every invocation; mesh-owned pointers are not retained across topology or storage changes.
- Patch selection is explicit and does not imply whole-field correction.
- Multiple writers may not target the same field at the same solver stage.

## Numerics

Derived operations require entries in fvSchemes. The default convenient policy keeps an exact user entry or a usable default and inserts only a missing requirement. Every insertion is recorded in numerics-changes.tsv. Strict numerics rejects the missing requirement instead.

## Atomic publication

All outputs are staged and validated before commit. A bad shape, conversion, non-finite value, or overlapping destination rejects the whole exchange. Boundary correction remains solver-owned unless an adapter explicitly defines another contract.

## External adapters

An external solver can include the installed headers, link the ABI-matched integration library, and register its equation hook without a FoamNordic source checkout. Directory discovery is available through:

    foamnordic dir --include
    foamnordic dir --openfoam-library

See the externalNutFjord example for the minimal build and registration shape.
