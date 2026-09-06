# Principles and compatibility

This document fixes the ownership boundaries and public compatibility policy
for FoamNordic 1.x. New convenience APIs and solver adapters must preserve
these rules.

## Purpose

FoamNordic is a native substrate for a-posteriori mathematical and
machine-learning operators in OpenFOAM. It gives model authors typed access to
solver-owned fields without putting Python, a model framework, or a general
workflow database inside the solver loop.

The primary success criterion is scientific integrity under repeated solver
calls. Every configured closure invocation completes one validated request and
one atomic response before the owning OpenFOAM equation advances.

## Ownership

- OpenFOAM owns meshes, fields, boundary conditions, dimensions, equations,
  discretisation, correction order, and physical time.
- A solver adapter selects an exact equation or correction boundary and binds
  current OpenFOAM storage to semantic tensor ports.
- Fjord owns transport and handshake mechanics, not field meaning.
- ModelHost owns FNOM loading, packing, scaling, inference, and model-local
  resources, not solver state.
- Longship owns placement, process lifecycle, logs, and results. It creates an
  execution plan; it is not a numerical data path.
- Python authors declarations and reads durable observations or results. It is
  not called once per OpenFOAM cell or closure exchange.
- Domain projects own domain equations and products. Combustion chemistry,
  flamelet tables, and complete custom solvers do not belong in the common
  runtime.

## Runtime invariants

1. Mesh and field views are rebound and validated at every invocation. Stable
   topology or address reuse is never assumed.
2. Repeated PIMPLE/PISO calls at one physical time remain distinct exchanges.
3. A response is committed only after every declared output passes its name,
   shape, type, time, and sequence checks. Partial field mutation is forbidden.
4. OpenFOAM ranks remain rank-local. A rank with an empty patch participates
   with a valid zero-length view instead of silently skipping the protocol.
5. Transport selection may change between SHM, UDS, UCX, and TCP without
   changing the scientific field contract.
6. FNOM is the executable model contract. Backend choice must not change the
   solver-facing declaration.
7. Failure is explicit and bounded. A missing ModelHost, stale response, model
   exception, or lost owner cannot let OpenFOAM continue with an unverified
   result.

## Public API policy for 1.x

The supported public surface consists of documented Python names, installed
C++ and OpenFOAM headers, the `FoamNordic::` CMake targets, FNOM schema, Rune
wire version, documented CLI commands, and documented dictionary keys.

- Backward-compatible additions are allowed in a minor release.
- A new optional argument must preserve the previous default behaviour.
- Renames require a documented deprecation path for at least one minor
  release. Silent semantic reinterpretation is not allowed.
- Removing a public symbol, changing a default scientific operation, changing
  tensor ordering, or changing field-commit timing requires a major release.
- FNOM and Rune readers reject unsupported newer schema or protocol versions;
  they do not guess.
- Native artifacts are tied to their declared platform and ABI. Compatibility
  across an OpenFOAM compiler, precision, label width, or release boundary is
  never inferred from a filename alone.
- Internal modules, generated build trees, private Python names, and files not
  installed by the SDK are not compatibility promises.

All breaking changes must appear in `CHANGELOG.md` with a migration example.
A deprecation warning must identify the replacement and the release in which
the old path may be removed.

## Non-goals

FoamNordic is not:

- a replacement for OpenFOAM equation ownership or boundary handling;
- a Python callback engine inside the solver loop;
- a training platform, experiment database, or SmartSim-style universal
  orchestrator;
- a bundled TensorFlow, PyTorch, detailed-chemistry, or combustion stack;
- a promise that every OpenFOAM turbulence model is reproduced internally;
- a global rank-zero field store;
- permission to conceal an unsupported field meaning behind automatic name
  matching.

Automatic bindings are conservative conveniences over an explicit contract.
An unknown or ambiguous semantic name must fail before launch and require the
user or adapter author to state the mapping.

## Extension rule

Generic transport, contracts, packing, observations, and lifecycle behaviour
belong in FoamNordic. Solver-specific equations and call sites belong in an
out-of-tree adapter. A proposed core change should therefore answer one
question: can two unrelated OpenFOAM solver projects reuse it without adopting
each other's physics? If not, it stays outside the core.

The official [out-of-tree adapter example](../../examples/openfoam/externalNutFjord/README.md)
is the minimum acceptance pattern for that boundary.
