# Changelog

This file records user-visible changes. Internal refactors and historical protocol revisions are intentionally omitted; the documentation describes the current FoamNordic contract only.

## [1.0.7] - 2026-09-10

### Model loading

- Joblib models containing enough small arrays to exhaust a platform's memory-mapping or file-table limit fall back to buffered loading. Ordinary models retain memory-mapped loading, while unrelated I/O failures remain visible.
- BindNordic now owns generic Joblib, compiled and Equinox loading and evaluation. FoamNordic's resident process retains worker orchestration, function-field adaptation and deterministic invocation context without duplicating backend runtime code.
- Model tensor views are independent of solver time and exchange metadata. FoamNordic keeps that context on Fjord tensors and supplies only model arrays across the BindNordic kernel boundary.

### Packaging and file I/O

- Model compilation, FNOM container handling, fitted-scaler normalization, backend payload serialization, and focused OpenFOAM field decoding are provided by BindNordic behind the existing FoamNordic API.
- FoamNordic no longer depends on foamlib for field discovery or postprocessing. OpenFOAM directive expansion remains tied to the case toolchain and is still handled by FoamNordic.
- Source-tree templates now live under `tools/templates`; packaged template discovery and the public tutorial API are unchanged.
- Native FNOM contracts, manifests, affine scaling, compiled and forest execution, and optional ONNX execution now live in BindNordic. FoamNordic keeps compatibility headers and owns transport, field packing, worker lifecycle and atomic publication.
- The default installation retains OpenFOAM execution and direct mathematical function support. `foamnordic[ml]` adds the optional model stack with CPU JAX, while `foamnordic[ml-cuda12]` selects CUDA 12 JAX explicitly on Linux.
- BindNordic is shipped inside the FoamNordic wheel rather than released as a separate distribution. The `ml` and `ml-cuda12` extras remain the only user-facing model-stack choices and avoid architecture-based GPU assumptions.

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
