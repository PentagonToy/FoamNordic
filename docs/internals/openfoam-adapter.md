# FoamNordic OpenFOAM adapter

The first native OpenFOAM adapter is intentionally thin. It is a
`fvMeshFunctionObject` named `foamNordicExchange`; scheduling remains an
OpenFOAM `controlDict` responsibility, while communication, atomic publication,
and output validation remain in the transport-independent FoamNordic core.

The supported baseline is OpenFOAM.com v2512 on Linux x86-64. OpenFOAM.com
v2606 on macOS arm64 is the development compatibility target. The adapter uses
only APIs common to those releases: `fvMeshFunctionObject`,
`lookupObjectRef`, and `primitiveFieldRef`.

## Field path

The OpenFOAM bridge supports internal `volScalarField`, `volVectorField`,
`volSphericalTensorField`, `volSymmTensorField`, and `volTensorField` storage.
Inputs become read-only native views over `primitiveField()` and outputs become
mutable views over `primitiveFieldRef()`. Python does not participate and the
field is not copied merely to cross the OpenFOAM/closure boundary. Rune/Harbor
transport owns any copy required by the selected channel.

Returned tensors remain staged until `AtomicFieldExchange` validates the full
commit. Only then are outputs copied into their solver-owned views and followed
by `correctBoundaryConditions()`. An input and output may name the same field,
so an identity or corrective `U -> U` contract remains atomic.

The bridge deliberately knows nothing about turbulence or combustion models.
Solver-integrated closures use the same views with `ClosurePort`, whose commit
blocks on every invocation. Specific OpenFOAM model adapters should only select
fields, expose their views, and invoke the port; transport, sequencing, scaler
application, bypass, and inference remain in the native core.

The distinction between generic field programs and equation-level adapters is
also recorded beside the implementations in
`src/foamnordic/openfoam/models/README.md`. Copied-case laminar, RAS, and LES
evidence is recorded in [OpenFOAM case validation](../benchmarks/openfoam-cases.md).

`Foam::foamNordic::ClosureSession` is the reusable solver hook. It owns one
rank-local connection and one `ClosurePort`; `begin(Time)` creates a fresh
per-call invocation without reconnecting. A turbulence correction, modeled
equation update, or combustion source evaluation provides its current views,
registers its outputs, and commits before returning to the solver. Repeated
calls at the same `Time::timeIndex()` therefore remain distinct transactions.
Rune v2 preserves that OpenFOAM index separately as `solver_time_index`; the
per-call `exchange_index` remains monotonic and unique for every outer
corrector or source evaluation.
The session also centralizes `{rank}` address expansion and UDS-to-SHM
negotiation so specific closure adapters cannot accidentally implement a
different handshake.

`Foam::foamNordic::ClosureHook` is the higher-level solver-facing boundary.
One call to `invoke(mesh, time)` creates a new `OperationFrame`, evaluates all
declared inputs from the current object registry, registers mutable output
views, blocks for one atomic closure commit, and corrects the returned field
boundaries before returning. A turbulence or combustion model therefore keeps
only a persistent hook and calls it at the exact native correction/source
evaluation site; it does not reproduce transport or feature plumbing.

## Operations at the solver call site

`Foam::foamNordic::operations::OperationFrame` evaluates derived inputs from
the active OpenFOAM `fvMesh` and object registry at the instant a closure is
invoked. Initial native operations are `grad`, `div`, `curl`, `laplacian`,
`mag`, `symm`, and `dev`, with nested expressions such as
`dev(symm(grad(U)))`. The implementation calls OpenFOAM `fvc` and field
algebra directly, so the case discretization schemes, dimensions, boundary
conditions, and current outer-corrector state remain authoritative.

One frame belongs to exactly one closure invocation. It caches repeated
subexpressions only within that invocation and owns all OpenFOAM `tmp` fields
until the blocking commit finishes. A later PIMPLE/PISO corrector constructs a
new frame and recomputes its features from the updated solver state. No Python
or external ClosureHost attempts to reconstruct OpenFOAM differential
operators.

Binary operations use the same expression tree. `dot(A,B)` and `ddot(A,B)`
apply OpenFOAM field contractions. `div(phi,U)` requires a registered
`surfaceScalarField`, while `laplacian(nu,U)` resolves `nu` from a registered
`volScalarField` or from `transportProperties`/`physicalProperties` as a
dimensioned scalar. Type-incompatible combinations fail before publication.

The development utility `foamnordicOpenFOAMOperationsProbe -case <path>`
loads `U` and `p` through the OpenFOAM registry and evaluates representative
scalar, vector, tensor, differential, nested, and contraction expressions. It
checks mesh-cell extent and every resulting scalar component for finiteness;
this is the required a-posteriori operation smoke test for each supported
OpenFOAM release.
The probe intentionally does not invent missing discretization entries. An
expression such as `div(U)` is exercised only by a case whose `fvSchemes`
defines `div(U)`; otherwise OpenFOAM's native missing-scheme failure remains
authoritative.

Solver hooks declare transport keys separately from OpenFOAM expressions:

```foam
foamNordicClosure
{
    address          "unix:///tmp/foamnordic-case-{rank}.sock";
    sessionId        2026;
    sharedMemory     true;
    ucx              false;

    inputs           (velocity_grad filter_width);
    inputExpressions ("grad(U)" delta);
    outputs          (nut);
}
```

When `inputExpressions` is omitted, every input key is also treated as its
primitive OpenFOAM field name. The two lists must otherwise have identical
lengths. Expressions are evaluated anew by each `ClosureHook::invoke()` call;
the cache never survives into a later PIMPLE/PISO outer corrector, even when
the physical time and `Time::timeIndex()` are unchanged.

A model adapter owns the hook for the model lifetime and invokes it directly
where the native closure is required:

```cpp
Foam::foamNordic::ClosureHook closure
(
    coefficients.subDict("foamNordicClosure")
);

// Inside correctNut(), a modeled-equation update, or a source evaluation:
const auto exchangeIndex = closure.invoke(mesh, mesh.time());
```

`invoke()` is intentionally blocking. Returning from it means that the full
output batch was validated, committed to the solver-owned internal fields,
and followed by boundary correction. Adapters must call it on every native
closure evaluation; they must not suppress repeated calls merely because the
OpenFOAM time index is unchanged.

### Optional native observation

A closure dictionary may declare a separate read-only observation stream:

```foam
observation
{
    path        "/path/to/run/observations.{rank}.jsonl";
    fields      (U p nut);
    every       100;
    offset      1;
    maxRecords  64;
    maxBytes    262144;
    overflow    dropOldest;
}
```

The block is optional. When absent, the OpenFOAM hook constructs no observation
plan, buffer, relay thread, or extra connection. When present, each rank
reduces its current internal fields after the atomic closure commit and
boundary correction. The solver thread performs only a non-waiting admission
to a byte-bounded ring; a separate writer flushes one JSONL record at a time.
The path supports `{rank}` expansion, so MPI ranks never contend for one file.
Python merges records sharing an exchange index into global min/max, weighted
mean, L2 norm, and count summaries. A failed writer disables only observation.
It cannot roll back, delay, or partially publish a closure result. The
lower-level native API also retains the framed Fjord publisher for applications
that need a live network observation stream.

`every` and `offset` refer to the monotonic closure exchange index, not the
OpenFOAM time index. Repeated PIMPLE/PISO corrections at one physical time are
therefore independently observable. Observation records contain `minimum`,
`maximum`, `mean`, `l2`, and `count`; full fields remain in OpenFOAM unless
a future explicit sampled-view product is declared.

### First solver-integrated LES model

`Foam::LESModels::nutFjord` is the first thin runtime-selected LES adapter. It
inherits OpenFOAM's `LESeddyViscosity`, owns one persistent `ClosureHook`, and
calls the hook from `correctNut()` on every native turbulence correction. The
adapter binds the active LES filter-width field as `delta`; `grad(U)` and other
features are still evaluated by `OperationFrame` at that exact call site. It
contains no socket, SHM, sequencing, scaler, bypass, or inference code.

The canonical coefficient body is
`src/foamnordic/template/openfoam/nutFjordCoeffs.in`. Its initial contract is
`grad(U), delta -> nut`. This deliberately establishes the smallest useful LES
closure boundary before k-equation and combustion adapters are added. A model
adapter may bind solver-owned fields that are not registered in the mesh, but
duplicate or empty binding names fail before publication.

FoamNordic registers the same templated adapter separately in OpenFOAM's
incompressible `turbulentTransportModels` and compressible
`turbulentFluidThermoModels` selection tables. Compressible solvers such as
`rhoPimpleFoam` still receive kinematic `nut`; OpenFOAM owns the density-based
conversion to dynamic turbulent viscosity and thermal diffusivity.

For native integration tests, `foamnordic_openfoam_echo` can map one input
tensor onto a differently named output without entering Python. For example,
the following publishes a zero-valued `nut` field with the shape of
`filter_width`:

```text
foamnordic_openfoam_echo \
    unix:///tmp/fjord-nut.sock \
    nut \
    0.0 \
    --source filter_width
```

The worker accepts repeated outer-corrector calls at one solver-time index and
monotonically advancing solver times, while rejecting regressing or
inconsistent metadata.

The ONNX fixture generator also emits `nutFjord.onnx` and `nutFjord.fnom`.
This native double-precision model consumes the packed
`velocity_grad(9), filter_width(1)` contract and publishes `nut(1)` using
`nut = 0.05 * filter_width`. It is intentionally simple, but it exercises the
same field packing, ONNX Runtime, output splitting, SHM publication, and
OpenFOAM field replacement path used by a trained LES closure.

`tools/openfoam/testNutFjordSolverSplitSlurm.sh` is the solver-integrated HPC
gate. It copies an LES case into scratch, replaces only its turbulence model
configuration with `nutFjord`, limits the run to three time steps, decomposes
the case, and launches parallel `pimpleFoam` through the central Longship Slurm
proxy. The original case remains unchanged. A pass requires runtime selection
of `nutFjord`, UCX negotiation for every rank, clean ONNX runner shutdown, the
requested final solver time, and coupled Longship completion.

This gate passed on Roihu with two OpenFOAM v2512 ranks and a resident
ClosureHost in separate Slurm allocations. Three `pimpleFoam` time steps
completed through the native `grad(U), delta -> nut` ONNX contract over UCX,
with clean rank, worker, scheduler-proxy, and Longship shutdown. This is the
baseline solver acceptance gate for later `kEqnFjord` case validation.

### One-equation LES model

`Foam::LESModels::kEqnFjord` extends the same native boundary to an SGS kinetic
energy equation. Its ordered contract is

```text
k, grad(U), delta -> nut, kProduction, kDissipationCoeff
```

The model retains OpenFOAM's native transient, convection, diffusion, source,
relaxation, constraint, and lower-bound operations. FoamNordic replaces only
the three closure quantities. One blocking exchange is performed during model
validation, then once before and once after every solved k equation. The model
contains no transport-specific code and reuses the same `ClosureHook`, atomic
publication checks, rank-local addressing, and SHM path as `nutFjord`.

Like `nutFjord`, `kEqnFjord` is registered for both incompressible and
compressible turbulence models. Its FNOM contract remains
`k, grad(U), delta -> nut, kProduction, kDissipationCoeff`; density,
dilatation, molecular diffusion, relaxation, and equation sources stay in the
OpenFOAM-owned k equation.

The canonical coefficient body is
`src/foamnordic/template/openfoam/kEqnFjordCoeffs.in`. A case selecting this
model must also provide the normal OpenFOAM `k` field and boundary conditions.

The native fixture generator emits `kEqnFjord.onnx` and `kEqnFjord.fnom` for
this ordered contract. Its deterministic matrix keeps the integration test
stable while exercising all packing and splitting paths:

```text
nut                   = 0.05 * delta
kProduction           = 0.02 * k
kDissipationCoeff     = 0.10 * k
```

The generator loads the completed artifact through ONNX Runtime and verifies
all three numerical outputs before publishing it for a solver test. The same
11-feature to 3-feature boundary also has a backend test independent of ONNX,
so field ordering and output splitting are checked on workstation builds.

`tools/openfoam/testKEqnFjordSolverSplitSlurm.sh` is the corresponding compact
solver gate. It reuses a prepared native/ONNX build, copies the source case,
selects `kEqnFjord`, runs two parallel `pimpleFoam` ranks against one resident
ClosureHost, and requires UCX, all three written closure fields, the requested
end time, and clean Longship shutdown. If the compact cavity has no `0/k`, the
driver seeds its known wall patches from `k.cavity.in`; a production case that
already owns `0/k`, such as NACA4412, retains its original field and boundary
conditions. The compact Smagorinsky source also has no k-equation numerics, so
the copied case receives an explicit `div(phi,k)` scheme. Its existing
`(U|k|epsilon|omega|R|nuTilda).*` solver entry already covers both `k` and
`kFinal` and is deliberately retained instead of being shadowed by generated
literal entries. Existing source dictionaries remain unchanged. This keeps the
cavity software gate distinct from later physical case validation.

The compact gate passed on Roihu with two OpenFOAM v2512 ranks in a `small`
allocation and one resident ClosureHost in an interactive allocation. Three
time steps exercised model validation plus the pre- and post-k-equation closure
updates; every rank used UCX, every expected field was written, and Longship
completed cleanly. The run establishes solver integration only. Its seeded 2D
field and maximum-iteration k solves are not reference settings for NACA4412.

A subsequent two-rank NACA4412 smoke passed with the case's existing mesh,
`0/k` boundary conditions, and PBiCG/DILU k solver. It exercised the same
three-output ONNX closure over UCX through three copied-case time steps and
completed under Longship. High Courant numbers and weak short-run pressure
residuals make this an interoperability result only; it does not validate the
fixture closure, the modified smoke numerics, or the aerodynamics.

The development executable `foamnordicOpenFOAMClosureHookProbe` reads
`system/foamnordicClosureDict` and performs two blocking calls at the same
OpenFOAM time index. Its reference result is evaluated independently with
an independent `OperationFrame`; the returned internal field must be an exact
identity match, and the two calls must receive consecutive exchange indices.
`probeExpression` and `probeOutput` select that development-only comparison,
so the same executable covers both a derived-field echo and an ONNX `U -> U`
fixture. `probeScale` additionally verifies non-identity field replacement;
for example, running `foamnordic_openfoam_echo <address> U 1.005` with
`probeScale 1.005` proves that the solver-owned velocity storage actually
changes in the C++ path. Together
with `foamnordic_openfoam_echo`, this tests expression evaluation, UDS-to-SHM
negotiation, per-call sequencing, atomic publication, output application, and
shutdown without a Python process.

`src/foamnordic/template/openfoam/closureDict.in` is the canonical source for
that dictionary. Generated cases substitute the address, session identity,
SHM policy, ordered transport keys, ordered OpenFOAM expressions, and output
fields without embedding those choices into a turbulence or combustion model.

For a development installation with OpenFOAM already loaded, the complete
native ONNX verification is reproducible with:

```bash
FOAMNORDIC_SOURCE_CASE=/path/to/case \
FOAMNORDIC_ONNX_RUNTIME_ROOT=/path/to/onnxruntime-1.28.0 \
tools/openfoam/testClosureHook.sh
```

The script builds in a unique work directory, copies both the case and wmake
sources, and leaves the repository untouched. Success requires all native
tests, an actual `laplacian(p)` ClosureHook input, exact `U * 1.005`
replacement, ONNX identity on two same-time OpenFOAM invocations, and an
explicitly observed SHM data plane. It also makes a worker reject one active
exchange and proves that the OpenFOAM field remains byte-identical. The
development probe seeds its output
field with a deterministic, non-zero in-memory pattern before the scaled and
ONNX checks. Verification therefore cannot pass vacuously and does not depend
on which time directories happen to be present in the source case. The seed is
never written to the copied case and is not part of the production adapter.
The echo worker also rejects skipped or repeated exchange indices,
inconsistent metadata within one exchange, and any physical-time change
between the two calls. This directly exercises the repeated outer-corrector
case instead of inferring it from final field values.
Copied wmake trees are cleaned before compilation, preventing objects,
dependency files, or `lnInclude` links from an earlier OpenFOAM release or
build directory from contaminating the verification.
The final report records SHA-256 digests for the OpenFOAM adapter, probe,
resident worker, ONNX graph, and native model manifest.
Each probe and worker receives a bounded 60-second completion window so a
broken blocking handshake fails at its named endpoint instead of hanging the
verification job. Set `FOAMNORDIC_TEST_TIMEOUT` to a positive number of
seconds when a slower debug build needs a larger window.

Each MPI rank resolves `{rank}` in its address independently. This preserves
rank-local fields and avoids a rank-zero gather. For example:

```text
unix:///tmp/foamnordic-case-{rank}.sock
```

Set `FOAMNORDIC_MPI_RANKS=2` (or a larger local rank count) when running
`tools/openfoam/testClosureHook.sh` to add a decomposed parallel probe. The
test starts one worker per rank, requires every rank to negotiate its own SHM
channel, and verifies two same-solver-time blocking calls without gathering
fields through rank zero.
It then launches one resident ONNX ClosureHost through Longship, connects every
OpenFOAM rank to that shared node-local listener, verifies one SHM session per
rank, and requires protocol shutdown to reap every runner before the coupled
workload reports success.
When Slurm grants one task with multiple CPUs, the test verifies that the CPU
count covers all requested ranks and supplies PRRTE's oversubscribe mapping
flag. This corrects PRRTE's slot accounting only; it does not launch more ranks
than the allocated CPU count.

## Dictionary

```foam
fjordClosure
{
    type            foamNordicExchange;
    libs            ("libfoamnordicOpenFOAM.so");
    enabled         true;
    executeControl  timeStep;
    executeInterval 1;
    writeControl    none;

    address         "unix:///tmp/foamnordic-case-{rank}.sock";
    sessionId       2026;
    sharedMemory    true;
    ucx             false;
    exchangeControl timeStep;

    inputs          (c_tilde c_var T_tilde);
    outputs         (omega_c);
}
```

`sharedMemory true` offers UDS and SHM during the blocking Rune handshake. If
the native worker accepts SHM, the established UDS connection is upgraded
before the first field exchange. TCP addresses remain on TCP and never pretend
to provide same-node SHM.

For a central host, use a TCP control address and request the UCX bulk path
explicitly:

```foam
address          "tcp://closure-host:24026";
sharedMemory     false;
ucx              true;
```

The adapter connects the TCP control session, requires the ClosureHost to
select UCX, and upgrades before publishing the first Rune tensor. Both the
CMake native libraries and the wmake adapter must be built with UCX support.
An explicit `ucx true` never falls back to TCP.

`exchangeControl timeStep` is the default and processes each OpenFOAM
`Time::timeIndex()` once. Re-entry at the same index—such as repeated PIMPLE
outer correctors—does not create a duplicate exchange. The opt-in `everyCall`
mode assigns a separate monotonic exchange index to every invocation while
retaining repeated physical time as independent metadata. Protocol identity
therefore never depends on PISO/PIMPLE iteration names.

## Native build contract

The wmake adapter expects:

- `FOAMNORDIC_SOURCE`: repository root;
- `FOAMNORDIC_BUILD`: CMake build containing the PIC native static libraries;
- `FOAMNORDIC_UCX_FLAGS`: optional UCX compile definition and include path;
- `FOAMNORDIC_UCX_LIBS`: optional UCX link and runtime-search flags;
- `FOAM_USER_LIBBIN`: OpenFOAM library destination.

The native libraries and adapter must be built with the compiler and ABI used
by the active OpenFOAM installation. A local v2606 compile and a Roihu v2512
compile are required before adding an a-posteriori performance case.

For that performance case, `foamnordic_openfoam_onnx_fixture <directory>`
creates a deterministic float64 `U -> Identity -> U` ONNX model and native
manifest. It is intentionally a development fixture: running a case against
it proves the complete OpenFOAM, SHM, resident ONNX and atomic field replacement
path while keeping the final velocity field byte-identical to the uncoupled
baseline.
