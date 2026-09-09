# Changelog

This file records user-visible changes. Internal refactors and historical protocol revisions are intentionally omitted; the documentation describes the current FoamNordic contract only.

## [1.0.6] - 2026-09-09

FoamNordic 1.0.6 consolidates the native runtime, lifecycle, artifact, and OpenFOAM integration work completed after the 1.0.5 baseline.

### Execution and memory

- Model requests announce their complete tensor count and payload size before transfer, allowing the worker to admit or reject the request atomically.
- Receive buffers, active execution storage, retained workspaces, and output publication have bounded ownership. Idle workspace storage is released when it would prevent an admissible request.
- Floating-point model outputs are checked before publication. A non-finite or malformed output leaves every OpenFOAM-owned destination unchanged.
- Runtime generations are immutable. Active runs pin the selected generation; cleanup preserves the current, recent, pinned, incomplete, and damaged generations.

### Reproducibility and artifacts

- `Longship(..., numerics="strict")` rejects a missing derived-field scheme. The default convenient mode adds only required schemes and records actual changes in `numerics-changes.tsv` inside the run directory.
- FNOM inspection identifies data-only and code-bearing payloads without executing them.
- `foamnordic diff` compares validated manifests and payload digests without loading the model.
- Sampled ModelHost timing is available through `Longship.launch(profile_every=N)`.

### Runtime and integration

- Native runtime builds are content-addressed and selected per OpenFOAM ABI.
- Local and Slurm runs share the same lifecycle, readiness, shutdown, logging, observation, and result contracts.
- Generic transforms and equation-level closures use the same field-program transport and model execution path.
- Derived OpenFOAM inputs support validated scalar algebra and square roots, allowing features such as strain-rate magnitude to be evaluated at the closure call site.
- Built-in LES adapters retain fixed solver-owned outputs while explicit FNOM inputs may bind to any compatible stored or derived field expression.
- `foamnordic dir --templates` exposes the packaged adapter scaffold without requiring a source checkout or an active OpenFOAM environment.

## [1.0.5]

FoamNordic 1.0.5 is the distribution baseline for CPython 3.11 and 3.12 on Linux x86_64, Linux aarch64, and Apple Silicon. Wheels contain the Python API, native control extension, and build kit. OpenFOAM and its matching compiler and MPI environment remain system dependencies.

Upgrade the Python package, restart long-lived Python processes, load the target OpenFOAM environment, and rebuild its integration:

```console
python -m pip install --upgrade foamnordic
foamnordic build
foamnordic doctor
```
