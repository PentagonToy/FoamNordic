# Changelog

This file records user-visible changes. Internal refactors and historical protocol revisions are intentionally omitted; the documentation describes the current FoamNordic contract only.

## [1.0.8] - Unreleased

### Tables and closure composition

- `fno.Export.table()` writes self-contained, data-only FNOM artefacts containing ordered regular-grid axes and dense output values. BindNordic evaluates them with native multilinear interpolation and explicit clamp or error bounds policies.
- `fno.Operator.table()` distinguishes lookup tables from executable model artefacts and verifies that distinction against the FNOM manifest before launch.
- Immutable `Closure.update()` declarations attach solver-mediated field updates to one physical closure without adding another top-level workflow abstraction. Each stage retains an independent artifact, kernel and transport endpoint, while the solver integration owns the call sites and any equations advanced between them.

### OpenFOAM integration

- Case validation no longer assumes conventional `U` and `p` initial fields. Solver-only cases validate their common directory and dictionary structure, while transforms, closures and observations validate the fields they actually declare; cases using fields such as `p_rgh` are therefore accepted without aliases.
- Case paths are canonicalised consistently before validation, including macOS paths that cross the `/tmp` to `/private/tmp` alias.
- `progressRateFjord` provides a thin native reaction-rate hook whose inputs remain explicitly configurable and whose solver-owned output is `omega_c`.
- Custom OpenFOAM integration templates can place closure update bodies explicitly, allowing a modelled reaction rate and a tabulated thermochemical-state update to share one `Longship` lifecycle without implying direct model composition.
- Runtime template discovery now follows one validated rule across OpenFOAM dictionaries, Slurm scripts and banners: packaged resources take precedence, with `tools/templates` used as the source-checkout fallback.

### Profiling

- Opt-in telemetry reports ModelHost workload, memory and stage timings together with per-rank solver request-to-response, request publication, response wait and atomic field-commit timings. `profile_every=0` leaves both execution paths uninstrumented.
- Packing exchange-validated active-cell indices no longer repeats a full validation scan; focused million-cell benchmarks are about 2.4–2.6 times faster and within 2% of the equivalent unchecked gather, without implying the same end-to-end solver speed-up.

### Resources

- Native runtime builds use the CMake-owned `foamnordic_runtime_bundle` target so every enabled runtime component, including the optional ONNX connector, is built before installation.
- FNOM execution metadata distinguishes backends that own compute threads from backends that support concurrent independent evaluations.
- Field programs assigned to the same node now run inside one ModelHost process and borrow the full available execution capacity from one shared pool instead of receiving static CPU partitions, while retaining separate artifacts, endpoints, kernels and backend state.
- `foamnordic build` installs ONNX support as a versioned runtime connector. A consolidated ModelHost loads it on demand, allowing ONNX and Python or table programs to share one process without linking the wheel extension to the host OpenFOAM toolchain.

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
