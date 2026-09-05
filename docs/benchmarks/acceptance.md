# Acceptance status

The solver-agnostic foundation has Linux HPC evidence for cross-node TCP,
fabric-backed UCX, attached ModelHost lifecycle, two OpenFOAM ranks,
`nutFjord`, and `kEqnFjord`. Domain-specific equation and physics acceptance
belongs to each solver repository; FoamNordic's acceptance matrix stops at the
adapter contract and exact field transport boundary.

The runnable entry points are in [`tools/hpc/`](../../tools/hpc/). They accept
site paths, account, partition, and environment through variables rather than
embedding a user or project identity.

## Multi-node acceptance

The attached multi-node software gate is runnable with
[`tools/hpc/testMultiNodeLongship.py`](../../tools/hpc/testMultiNodeLongship.py).
It runs the same short, two-node decomposition first without a field program
and then with an identity `Transform`. Passing requires both runs to succeed
and their final `U` and `p` arrays to agree within `1e-12`. This is a transport
and lifecycle gate, not a scaling benchmark.

Roihu's two-node `medium` gate passed on 2026-08-31 with OpenFOAM v2512.
Baseline job `962069` ran on `rc4231-4232` in 6 seconds; identity-coupled job
`962071` ran on `rc4152-4153` in 8 seconds. Both final fields matched exactly:
maximum absolute error was `0.0` for `U` and `p`. This validates one attached
ModelHost per node, node-local SHM exchange, distributed MPI identity, and
coupled startup/shutdown. It is not a cross-node scaling measurement because
the two paths ran on different node pairs.

## Deliberately deferred

Cross-node process placement is validated independently by the two-node
identity gate below.
It does not need one two-node allocation: on a site with one-node development
partitions, the preferred topology is a resident ModelHost in an existing
`interactive` allocation and a one-node solver job submitted to `small`,
excluding the host node. This mirrors the already validated UCX lifecycle
gates. It is kept out of the default suite so the first failure can be
attributed to combustion or MPI before adding cross-allocation placement.

Large multi-node scaling and central accelerator placement remain later
campaigns. They should use deliberately requested partitions and report
solver, model, and orchestration time separately.

Production flamelet thermodynamics, conservation tolerances, realizability,
and a physically representative trained artifact remain scientific acceptance
work. They are not replaced by the software gates above.

## Roihu evidence: 2026-08-23

OpenFOAM v2512 on one `small` node validated the reduced 4000-cell
counter-flow case. Job `790448` established the four serial gates at commit
`600d1a5`; job `790497` completed the focused two-rank MPI gate at commit
`56338b4` after run-local decomposition normalization was added.

| Gate | Ranks | Gate elapsed | Result |
| --- | ---: | ---: | --- |
| Stock `reactingFoam`, volumetric rate | 1 | 2.444 s | PASS; source `1.0e-4 kg/m3/s` |
| Stock `reactingFoam`, specific rate | 1 | 2.378 s | PASS; source `1.0e-4 1/s` |
| Progress coordinator, volumetric rate | 1 | 2 s | PASS; three time steps |
| Progress coordinator, specific rate | 1 | 2 s | PASS; three time steps |
| Progress coordinator, specific rate MPI | 2 | 3 s | PASS; three time steps |

All three coordinator paths reported the same displayed ranges:
`c_tilde = 2.45017e-1 .. 2.62339e-1` and
`omega_c = 1.47532e-2 .. 1.50997e-2`. The reported maximum error in
`CH4 + CO2 = 1` was `0.0` in serial and MPI. The first MPI attempt exposed a
test-harness mismatch between two requested ranks and a copied four-domain
hierarchical decomposition; it failed before solver launch and is not counted
as physics or transport evidence.

Job `790528` then ran the same mesh and `endTime=1` with stock `reactingFoam`
only: no FoamNordic import, runtime library, or resident worker. The mesh
passed `checkMesh`, the solver reached `End`, and OpenFOAM reported
`ExecutionTime = 0.08 s`; the corresponding FoamNordic stock-adapter solver
logs reported `0.09 s`. These single-step timings have only 0.01-second log
resolution, so they demonstrate comparable solver cost rather than a stable
percentage overhead. The FoamNordic gate totals of 2.378--2.444 seconds also
include artifact export, case preparation, and resident startup/shutdown.

A strict identical-mesh comparison at physical time 1 gave the following
largest errors against pure OpenFOAM:

| Candidate | `U` max abs / relative L2 | `T` max abs / relative L2 | `CH4` max abs / relative L2 |
| --- | --- | --- | --- |
| Volumetric-rate adapter | `1.0e-6 / 1.60e-7` | `1.0e-2 / 1.60e-7` | `1.0e-6 / 8.02e-7` |
| Specific-rate adapter | `1.0e-6 / 8.70e-8` | `0 / 0` | `1.0e-7 / 1.15e-7` |

Stored `p`, `O2`, `CO2`, and `H2O` were identical in both comparisons.
The remaining differences are at or near the case's six-digit ASCII output
precision; this is no-op/adapter compatibility evidence, not a scientific
validation of the placeholder reaction model.
