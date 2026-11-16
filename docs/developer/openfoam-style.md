# OpenFOAM source style

Use OpenFOAM conventions for code compiled against its API. The formatter covers `.C` and `.H` files in `src/foamnordic/openfoam/`, `tools/openfoam/` and `examples/openfoam/`. The standalone runtime, transport, backend, bindings, benchmarks and tests follow FoamNordic's C++ conventions.

## Format and check

Run these commands from the repository root with `clang-format` available:

```bash
tools/style/formatOpenFOAM.sh --check
tools/style/formatOpenFOAM.sh --apply
```

Use `--check` to report formatting differences and `--apply` to update the scoped files. Review the diff after applying [the repository configuration](../../tools/style/openfoam.clang-format). Use the formatter's layout for wrapped conditions and signatures. Preserve file banners, licence notices, `//-` member descriptions and meaningful section comments.

Use the closest source or tutorial from the target OpenFOAM release when reviewing field types, runtime keys or case dictionaries. The primary HPC target is OpenFOAM.com v2512. A case banner can record the release from which that file came. Use the [installation guide](../installation.md) to identify the runtime for the target platform.

Keep style changes separate from changes to types, public APIs, numerical expressions, field dimensions, boundary conditions and solver operation order. For case changes, verify the `FoamFile` object and follow the [verification guide](verification.md) for any parser, build or solver checks.
