# Changelog

Release notes are maintained here and reused verbatim as GitHub release bodies.
Version 1.0.5 establishes the documented distribution baseline; its feature
summary consolidates earlier development rather than claiming every feature
was introduced in this patch. Publication dates are recorded by GitHub releases.

## Unreleased

### Native model execution

- Added the Smedja model-execution boundary. Immutable tensor layouts are
  compiled once, while tensor addresses and shapes are rebound and validated
  for every invocation so dynamic OpenFOAM storage remains safe.
- Moved field-name lookup outside the hot cell loop and added a zero-copy
  ownership transfer for compatible single-output staging buffers.
- Narrowed artifact-kernel serialization to backend evaluation; request-local
  packing, scaling, validation, and staging no longer share the worker lock.
- Reused rank-local input workspace capacity across stable invocations, with
  automatic downsizing after large dynamic-mesh requests.
- Added a contiguous-cell Smedja packing path while preserving the general
  sparse gather path and per-invocation topology validation.
- Reused packed backend staging for the widest field during multi-output
  unpacking, removing one full-field allocation without exposing aliases.
- Allowed floating-point solver/model dtype differences to be converted during
  Smedja packing, with a specialized three-scalar float64-to-float32 path.
- Verified that changing payload sizes, including tensors larger than one SHM
  slot, stream through one session without unsafe shared-region replacement.
- Added internal surface-field dispatch alongside the existing volume-field
  bridge and accepted zero-cell rank-local exchanges as valid no-op batches.
- Added first-class boundary-patch metadata for closures and transforms.
  Patch storage is rebound from the current mesh for every invocation, empty
  rank-local patches remain valid, and explicit patch outputs are not replaced
  by an implicit whole-field boundary correction.
- Added an OpenFOAM `dynamicFvMesh` conformance probe that updates the mesh
  before each patch exchange and verifies repeated rank-local SHM rebinding.
- Verified the moving-mesh probe under MPI with a zero-face patch on one rank
  and a non-empty copy of the same patch on its peer.
- Fixed fresh `foamnordic build --without-onnx` installations by building the
  always-installed native inference core independently of the ONNX ModelHost.

### Solver boundary

- FoamNordic now builds and installs only the solver-agnostic native runtime,
  ModelHost, field hooks, and OpenFOAM adapter SDK. The bundled
  progress-variable solver and combustion-model registrations moved out of
  the common runtime and are owned by their domain solver project.
- `Longship` uses one closure/transform contract for stock and custom solvers;
  the progress-variable-specific `combustion=` branch was removed.
- `foamnordic dir --runtime`, `--include`, and `--openfoam-library` provide
  scriptable ABI-specific SDK paths, allowing external solver adapters to
  build without a FoamNordic source checkout.

## [1.0.5]

FoamNordic 1.0.5 establishes the reference release baseline for packaging,
installation, and the documented Python API. FoamNordic remains active research
software; this baseline is not a guarantee for every OpenFOAM or HPC environment.

### Baseline capabilities

- A common `Longship` interface for baseline OpenFOAM cases, field transforms,
  equation-level closures, and progress-variable combustion workflows.
- Self-contained FNOM artifacts with input/output tensor contracts, scaling,
  and runtime metadata. Export paths cover ONNX, sklearn-compatible estimators
  (supported compiled models or Joblib), and Equinox models.
- Local and Slurm execution, model placement, runtime observations, and result
  collection, with generic field-program inference separated from closure hooks.
- ABI-specific persistent OpenFOAM runtimes built with `foamnordic build` and
  environment diagnostics through `foamnordic doctor`.

### Packaging and runtime improvements

- Wheels include the complete native build kit, including Longship sources.
  Wheel CI compiles the installed runtime, adapter, inference library, and
  Longship executable and checks its command-line entry point.
- Native wheel distribution for CPython 3.11 and 3.12 on Linux x86_64,
  Linux aarch64, and macOS arm64. Wheels provide the Python package, native
  control extension, and build kit; the OpenFOAM integration is built locally.
- Persistent runtimes include the Longship executable and retain custom
  OpenFOAM runtime libraries.
- Native builds respect the selected OpenFOAM compiler environment when
  isolated Python environments introduce competing compiler paths.
- Version consistency and changelog extraction checks accompany wheel
  preparation. Release notes come from this changelog, not a separate summary.

### Compatibility and upgrade notes

- Earlier development names that treated all inference as closures have been
  removed. Update integrations to the documented field-program/model-host API;
  removed compatibility aliases are not restored by this release.
- Upgrade the Python package and its `_native` extension together, then rebuild
  the OpenFOAM runtime. `foamnordic build` is not a Python-package reinstall.
  Restart Python processes and notebook kernels after upgrading.
- For a regular virtual environment:

  ```bash
  python -m pip install --upgrade foamnordic==1.0.5
  # Activate your supported OpenFOAM/compiler environment first.
  foamnordic build
  foamnordic doctor
  ```

- Tykky/container users must ensure the launcher selects matching Python and
  native packages. Site-specific installation and migration instructions live
  in [CSC-HPC-Guide](https://github.com/PentagonToy/CSC-HPC-Guide/tree/main/python-environment).
  Changes to that installer are separate from this PyPI release.
- OpenFOAM, its matching compiler/MPI environment, and optional execution
  dependencies are not supplied by the Python wheel. This release does not
  distribute Windows wheels or source distributions.
