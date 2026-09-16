# OpenFOAM source style

Code compiled directly against the OpenFOAM API follows OpenFOAM's source layout so solver developers can read and extend adapters without translating between visual conventions. This rule applies to `src/foamnordic/openfoam/`, OpenFOAM-linked `.C` and `.H` files under `tools/openfoam/`, and `examples/openfoam/`.

Standalone runtime, transport, backend, binding, benchmark and test code retains FoamNordic's modern C++ style. Style does not change component ownership: OpenFOAM continues to own equations, mesh state, correction order, boundary conditions and field commits, while FoamNordic owns exchange and resident execution.

## Format and check

Run the scoped formatter from the repository root:

```bash
tools/style/formatOpenFOAM.sh --apply
tools/style/formatOpenFOAM.sh --check
```

The script uses [`openfoam.clang-format`](../../tools/style/openfoam.clang-format) and an explicit source-root allowlist. It does not traverse generated build trees or standalone native code. Formatting changes must remain presentational: do not replace STL types with OpenFOAM containers, rename public APIs, change numerical expressions, reorder solver operations or introduce additional OpenFOAM dependencies as part of a style pass.

OpenFOAM's hand-written layout includes conventions that are not fully expressible through `clang-format`. Preserve established `//-` member documentation, section comments and readable stream expressions when editing a formatted file; prefer the nearest corresponding source in the configured OpenFOAM reference tree when a construct is ambiguous.
