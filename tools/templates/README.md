# FoamNordic templates

Templates are grouped by the component that consumes them:

- `openfoam/` contains dictionaries and dictionary fragments copied into an isolated case. `derivedSchemes.json` contains conservative fallback schemes for ML input expressions; they are added only when neither an exact scheme nor a usable section default exists in the copied case. Its `model-adapter/` directory is a deliberately non-compilable scaffold for a new equation-level OpenFOAM closure. Domain-specific equation and model templates belong to the solver project that owns those equations.
- `slurm/` contains scheduler scripts. Solver output belongs in `logs/Sailing_<name>_<jobid>.out`; FoamNordic lifecycle output belongs in `logs/Sailing_<name>_<jobid>.log`.
- `shell/` contains local execution and environment wrappers.
- `large_banner.txt` is the runtime splash shown at the start of a user-facing FoamNordic run. Do not paste it into source files, generated dictionaries, or routine diagnostic output.
- `small_banner.txt` is the file header for FoamNordic-owned C and C++ sources that do not depend directly on the OpenFOAM API. This includes the native runtime, communication, inference, bindings, tests, and standalone tools.
- `openfoam/openfoamHeader.in` is the file header for C++ sources and headers compiled against OpenFOAM, including `wmake` libraries, turbulence models, function objects, probes, and out-of-tree adapter examples. Keep the short FoamNordic licence block immediately below it. Do not use the small banner in these files.

New FoamNordic-authored files use `SPDX-License-Identifier: GPL-3.0-only`. Preserve an upstream file's original copyright and licence notice when it is copied or substantially derived; do not replace such a notice with a FoamNordic template. Generated OpenFOAM dictionaries may use `openfoamHeader.in` without a source-code licence block when a conventional `FoamFile` header is useful.

Use `@UPPER_SNAKE_CASE@` for substitutions. Every token must be resolved before the generated file is executed. Keep site-specific modules, accounts, partitions, absolute paths, model names, and field contracts out of templates; they belong in the Python declaration or runtime profile. Add a new template when a solver family needs a structurally different dictionary instead of growing conditional string assembly in the launcher.

Repository examples and generated templates use placeholders such as `<allocation-account>`, `<user>`, `<jobid>`, and `<compute-node>`. Do not commit personal names, email addresses, project numbers, home or scratch paths, node names, or fabric addresses. Package-author metadata is the deliberate exception and belongs only in `packages/foamnordic/pyproject.toml`.

Templates may describe pure OpenFOAM or model-coupled runs. A template must not silently enable a closure, change physical boundary conditions, or mutate the source case.

Observation paths must contain `{rank}` whenever a case may run with multiple MPI ranks. They are generated below the marked run directory so `foamnordic clobber --workspace ...` can remove the whole owned run without guessing which files belong to FoamNordic.
