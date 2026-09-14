# Changelog

This changelog records user-visible changes in reverse chronological order. Internal refactors and superseded protocol revisions are omitted; the documentation describes the current FoamNordic contract.

## [1.0.8] - 2026-09-14

### Documentation and development

- `foamnordic build` now opens with the packaged FoamNordic runtime banner and a compact native-build heading.
- User-facing tables and tutorials now use the Onsaemiro 1.1 `Table` API, and the installation contract requires `onsaemiro>=1.1.2` for Rich terminal tables and publication-style Jupyter output.
- The Python reference now records public signatures, defaults, accepted values, lifecycle results, observation records, numerical execution lanes, FNOM metadata, post-processing metrics and command-line options in one self-contained documentation set.
- Architecture and benchmark records now use explicit ownership, lifecycle, measurement and interpretation contracts with consistent terminology and rendered diagrams.
- Repository guides now use consistent purpose, ownership, usage and verification sections across packages, native components, examples, tutorials, templates and release tooling.
- A developer guide now maps exchanges, closures, OpenFOAM adapters, observations, FNOM and ModelHost backends to their owning source directories and verification gates.
- Tutorial notebooks now use the current native `kEqnFjord` contract, explicit public API arguments and descriptive shell-safe filenames. The current Extra Trees and Equinox DNN examples train from one downloadable LES dataset, launch through Slurm and display live `Observe` summaries; the basic lid-driven-cavity baseline remains local.
- The `ml` and `ml-cuda12` authoring extras now include Optax so the Equinox training tutorial does not require an undeclared optimiser dependency.
- The repository root is a uv workspace containing the FoamNordic package and its virtual BindNordic source component; the published wheel remains one `foamnordic` distribution containing both namespaces.

### Tables and closure composition

- Uniform table axes that collapse at binary64 precision are rejected during native loading instead of producing a non-finite interpolation fraction.
- Verbose table exports identify their payload as `Table + FNOM` rather than `ONNX + FNOM`.
- FNOM table images store exactly uniform axes as start, step and count while retaining explicit non-uniform axes; the native evaluator accepts both the earlier and current table encoding.
- `fno.Export.table()` writes self-contained, data-only FNOM artefacts containing ordered regular-grid axes and dense output values. BindNordic evaluates them with native multilinear interpolation and explicit clamp or error bounds policies.
- `fno.Operator.table()` distinguishes lookup tables from executable model artefacts and verifies that distinction against the FNOM manifest before launch.
- Immutable `Closure.update()` declarations attach solver-mediated field updates to one physical closure without adding another top-level workflow abstraction. Each stage retains an independent artefact, kernel and transport endpoint, while the solver integration owns the call sites and any equations advanced between them.

### OpenFOAM integration

- GPU-backed Slurm launches now assign the requested GPU only to ModelHost and explicitly prevent the concurrent OpenFOAM step from inheriting that generic resource.
- Native graph eligibility now examines traced outputs as well as declared ports, so hidden patch references remain resident, every lowered differential operator contributes its required `fvSchemes` entry, and ambiguous expressions that mix pre-solve and post-solve `k` aliases are rejected during declaration.
- Native algebra publication snapshots and prevalidates every output field type, dimension and finite internal value before writing, publishes all internal fields before boundary correction, and supports the same `Observe` and sampled solver-telemetry controls as resident transforms.
- `Field.transport()` allows a mathematical `Operator.function` to declare one implicit scalar solve. `kEqnFjord` compiles the graph once, executes its finite-volume equation entirely inside OpenFOAM, and exposes the same fixed `k` and `nut` outputs as a directly learned state update.
- Fully symbolic `nutFjord` functions are compiled once and evaluated at the native OpenFOAM correction boundary without starting a ModelHost or exchanging fields. Unsupported functions retain the resident execution path.
- Fully symbolic field `Transform` functions use the same ModelHost-free path at supported time-step boundaries, while model-backed transforms retain their existing resident execution contract.
- Case validation no longer assumes conventional `U` and `p` initial fields. Solver-only cases validate their common directory and dictionary structure, while transforms, closures and observations validate the fields they actually declare; cases using fields such as `p_rgh` are therefore accepted without aliases.
- Case paths are canonicalised consistently before validation, including macOS paths that cross the `/tmp` to `/private/tmp` alias.
- `progressRateFjord` provides a thin native reaction-rate hook whose inputs remain explicitly configurable and whose solver-owned output is `omega_c`.
- Custom OpenFOAM integration templates can place closure update bodies explicitly, allowing a modelled reaction rate and a tabulated thermochemical-state update to share one `Longship` lifecycle without implying direct model composition.
- Runtime template discovery now follows one validated rule across OpenFOAM dictionaries, Slurm scripts and banners: packaged resources take precedence, with `tools/templates` used as the source-checkout fallback.

### Profiling

- Opt-in telemetry reports ModelHost workload, memory and stage timings together with per-rank solver request-to-response, request publication, response wait and atomic field-commit timings. `profile_every=0` leaves both execution paths uninstrumented.
- Packing exchange-validated active-cell indices no longer repeats a full validation scan; focused million-cell benchmarks are about 2.4–2.6 times faster and within 2% of the equivalent unchecked gather, without implying the same end-to-end solver speed-up.

### Resources

- `foamnordic doctor` warns when the Python package and native extension come from different installations, making stale source and wheel combinations visible before execution.
- Native FNOM payloads use a model-owned read-only mapping during loading, avoiding an initial whole-payload file copy and providing a safe lifetime for direct native views.
- Embedded ONNX payloads retain that mapping while the native session is active, avoiding an extra copy and restoring compilation of the mapped loading path.
- Resource summaries count only resident field programmes, matching the lowered launch plan so a native-only workload reports no ModelHost CPU, memory, shared memory or process allocation.
- Backend capabilities declare memory residence, completion mode, persistent device state, shape-sensitive workspace recreation, preferred batch rows and alignment. A residency planner distinguishes direct, explicitly staged and unsupported paths; unsupported execution fails before model evaluation rather than copying data implicitly.
- A transactional resident-engine contract supports long-lived state, optional checkpoint and restore, explicit topology-change behaviour and opt-in cost observation without adding domain semantics or mesh redistribution to the runtime.
- Native runtime builds use the CMake-owned `foamnordic_runtime_bundle` target so every enabled runtime component, including the optional ONNX connector, is built before installation.
- FNOM execution metadata distinguishes backends that own compute threads from backends that support concurrent independent evaluations.
- Field programs assigned to the same node now run inside one ModelHost process and borrow the full available execution capacity from one shared pool instead of receiving static CPU partitions, while retaining separate artefacts, endpoints, kernels and backend state.
- `foamnordic build` installs ONNX support as a versioned runtime connector. A consolidated ModelHost loads it on demand, allowing ONNX and Python or table programs to share one process without linking the wheel extension to the host OpenFOAM toolchain.
- Native tables use the managed resident path and remain available after `foamnordic build --without-onnx`; the ONNX connector now derives its scheduling capabilities from the same declaration as the underlying BindNordic kernel.

## [1.0.7] - 2026-09-10

### Model loading

- Joblib models containing enough small arrays to exhaust a platform's memory-mapping or file-table limit fall back to buffered loading. Ordinary models retain memory-mapped loading, while unrelated I/O failures remain visible.
- BindNordic now owns generic Joblib, compiled and Equinox loading and evaluation. FoamNordic's resident process retains worker orchestration, function-field adaptation and deterministic invocation context without duplicating backend runtime code.
- Model tensor views are independent of solver time and exchange metadata. FoamNordic keeps that context on Fjord tensors and supplies only model arrays across the BindNordic kernel boundary.

### Packaging and file I/O

- Model compilation, FNOM container handling, fitted-scaler normalisation, backend payload serialisation, and focused OpenFOAM field decoding are provided by BindNordic behind the existing FoamNordic API.
- FoamNordic no longer depends on foamlib for field discovery or postprocessing. OpenFOAM directive expansion remains tied to the case toolchain and is still handled by FoamNordic.
- Source-tree templates now live under `tools/templates`; packaged template discovery and the public tutorial API are unchanged.
- Native FNOM contracts, manifests, affine scaling, compiled and forest execution, and optional ONNX execution now live in BindNordic. FoamNordic keeps compatibility headers and owns transport, field packing, worker lifecycle and atomic publication.
- The default installation retains OpenFOAM execution and direct mathematical function support. `foamnordic[ml]` adds the optional model stack with CPU JAX, while `foamnordic[ml-cuda12]` selects CUDA 12 JAX explicitly on Linux.
- BindNordic is shipped inside the FoamNordic wheel rather than released as a separate distribution. The `ml` and `ml-cuda12` extras remain the only user-facing model-stack choices and avoid architecture-based GPU assumptions.

## [1.0.6] - 2026-09-09

FoamNordic 1.0.6 consolidates the native runtime, lifecycle, artefact and OpenFOAM integration work completed after the 1.0.5 baseline.

### Execution and memory

- Model requests announce their complete tensor count and payload size before transfer, allowing the worker to admit or reject the request atomically.
- Receive buffers, active execution storage, retained workspaces, and output publication have bounded ownership. Idle workspace storage is released when it would prevent an admissible request.
- Floating-point model outputs are checked before publication. A non-finite or malformed output leaves every OpenFOAM-owned destination unchanged.
- Runtime generations are immutable. Active runs pin the selected generation; cleanup preserves the current, recent, pinned, incomplete, and damaged generations.

### Reproducibility and artefacts

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
