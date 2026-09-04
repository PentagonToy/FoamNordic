# Changelog

Release notes are maintained here and reused verbatim as GitHub release bodies.
Version 1.0.5 establishes the documented distribution baseline; its feature
summary consolidates earlier development rather than claiming every feature
was introduced in this patch. Publication dates are recorded by GitHub releases.

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
