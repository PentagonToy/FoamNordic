# HPC transport status and site notes

This page is the operational summary for FoamNordic transports on HPC systems.
It distinguishes three different claims:

1. the site provides the required operating-system or communication runtime;
2. FoamNordic implements the corresponding Fjord channel;
3. the complete path has passed a representative end-to-end test.

An installed library, or an MPI build that uses it internally, does not by
itself satisfy the second or third claim.

## Current status

| Data plane | Intended placement | FoamNordic implementation | Example HPC validation |
| --- | --- | --- | --- |
| UDS | Same node, separate processes | Complete | PASS: OpenFOAM and native ONNX worker over forced pure UDS |
| SHM | Same node, solver rank and ClosureHost | Complete | PASS: two OpenFOAM ranks, shared resident ONNX ClosureHost, and Longship lifecycle |
| TCP | Different nodes | Complete | PASS: split Slurm allocations on two compute nodes |
| UCX | Different nodes, high-performance bulk transfer | UCP stream channel and TCP-to-UCX upgrade complete | PASS: split-allocation OpenFOAM field exchange with a resident ONNX ClosureHost |

The safe automatic policy is therefore:

```text
same node       UDS rendezvous -> SHM bulk when negotiated
same node       UDS data path when SHM is disabled or unavailable
different node  UCX when its build and site probe are verified
different node  TCP otherwise
```

UCX is now an HPC-validated Fjord data plane. The optional `UcxChannel` links
directly to the UCP API and upgrades a TCP Harbor before its first payload.
Automatic mode may select it only when the UCX build is present and the site
connectivity probe has passed. An explicit UCX request must fail before the
first exchange if it cannot be honored; it must never silently become TCP.
The native ClosureHost and OpenFOAM adapter expose an explicit UCX upgrade.
Their split-allocation integration gate has passed on a Slurm HPC system: an actual
OpenFOAM probe on a `small` node exchanged fields with the resident ONNX
ClosureHost in an interactive allocation.

## HPC validation record (CSC Roihu example)

These results were observed on 2026-08-21 on CSC Roihu. The allocation account,
user identity, job identifiers, node names, and fabric addresses are redacted.
They are site evidence, not portable performance promises.

### UDS

A Slurm job ran an actual OpenFOAM case and resident ONNX worker with
the SHM upgrade disabled. The worker reported `Data plane: UDS`, inference and
field exchange passed, and the endpoint was cleaned up.

### SHM

Another Slurm job connected two OpenFOAM ranks to one shared native ONNX
ClosureHost on the node. Both rank sessions negotiated SHM, performed atomic
field replacement, and completed under Longship supervision. The broader
native OpenFOAM hook also passed independently.

SHM uses UDS for rendezvous and control. Linux waits through process-shared
futex words; macOS retains the connected UDS as its blocking wake path. The
bulk Rune frames remain in POSIX shared memory.

### TCP

The reusable split-allocation driver is
`tools/network/testTcpSplitSlurm.sh`. It keeps a server in an existing
one-node interactive allocation and submits a one-node `small` client while
excluding the server node. This avoids waiting for a scarce full two-node
allocation.

The server ran in an interactive allocation and the client ran in `small` on
a different node. The probe completed 100 atomic round trips with a 1 MiB payload in
each direction per exchange:

| Result | Value |
| --- | ---: |
| Round trips/s | 21.5159 |
| Bidirectional payload MiB/s | 43.0318 |
| Server | PASS |
| Client | PASS |

The `argos-vars` warning emitted while submitting from the interactive compute
node disabled an auxiliary CSC integration for that job; it did not prevent
Slurm execution or invalidate the TCP transport result.

### UCX environment

Loading `openfoam/2512` on Roihu also loaded a compatible compiler, Open MPI,
and UCX stack:

| Component | Observed configuration |
| --- | --- |
| UCX tools | `ucx_info` and `ucx_perftest` available |
| UCX | 1.20.0, headers and shared/static libraries enabled |
| Open MPI | 5.0.10, configured with UCX |
| Open MPI components | `pml:ucx` and `osc:ucx` present |
| CPU threading | UCX multi-thread support enabled |
| Shared-memory support | CMA and KNEM enabled; XPMEM disabled |
| Network support | RC, UD, DC, mlx5, RDMA CM, and verbs enabled |
| Accelerator support | CUDA, ROCm, and GDRCopy disabled |

Roihu can build and run the CPU-only UCP client. Open MPI's `pml:ucx` transports
MPI traffic independently; the result below verifies FoamNordic's own Fjord,
Rune, and Harbor path rather than relying on that MPI component.

### UCX two-node validation

The interactive server and one-node `small` client ran on different nodes.
TCP established the Harbor control
session, after which the server advertised its `ib0` address and both peers
upgraded to a UCP stream. The probe completed 100 atomic round trips with one
1 MiB tensor sent and returned per exchange:

| Result | Value |
| --- | ---: |
| Round trips/s | 1647.19 |
| Bidirectional payload MiB/s | 3294.38 |
| Server | PASS |
| Client | PASS |

The run set `UCX_TLS=rc,ud,sm,self`, excluding UCX's TCP transport. Roihu
reported RC and UD devices on `mlx5_0:1` and `mlx5_1:1`; therefore this PASS is
not a UCP-over-TCP fallback. The client printed `Data plane: UCX`, and Rune
metadata, payload, atomic completion, monotonic exchange identity, and clean
shutdown were all verified. The CSC `argos-vars` submission warning again
disabled only the auxiliary ARGOS integration and did not affect the job.

### Building the optional channel

The feature is deliberately opt-in:

```bash
cmake -S . -B build-ucx \
    -DCMAKE_BUILD_TYPE=Release \
    -DFOAMNORDIC_UCX=ON \
    -DFOAMNORDIC_UCX_ROOT="$UCX_ROOT" \
    -DFOAMNORDIC_NETWORK_TOOLS=ON
cmake --build build-ucx --parallel
ctest --test-dir build-ucx --output-on-failure
```

The UCX-enabled Fjord unit test performs a loopback TCP negotiation, upgrades
to a UCP stream, and checks a field round trip. The inter-node acceptance test
then uses `tools/network/testUcxSlurm.sh` in one two-node allocation or
`tools/network/testUcxSplitSlurm.sh` across interactive and `small`
allocations. A successful client must explicitly print `Data plane: UCX`.

### Central OpenFOAM and ONNX acceptance

`tools/openfoam/testClosureHook.sh` can extend its normal UDS/SHM regression
suite with the central UCX gate. Run it inside the one-node interactive
allocation, set `FOAMNORDIC_UCX_SPLIT=true`, and provide the UCX installation
prefix. The script builds the native libraries and wmake adapter with the same
headers, starts one resident ONNX ClosureHost locally, then submits a one-node
`small` OpenFOAM client while excluding the host node.

The acceptance run forces `UCX_TLS=rc,ud,sm,self` by default, so UCP cannot use
its TCP transport. Success requires all of the following: the worker reports
`data plane: UCX`, the OpenFOAM hook reports `PASS`, the Slurm client completes,
and the host readiness marker disappears. Failed or interrupted drivers cancel
the submitted client and terminate the host.

The central acceptance gate passed on 2026-08-21. An interactive allocation
hosted the resident ONNX ClosureHost; a `small` job on another node ran the
OpenFOAM v2512 closure-hook probe. TCP established control, after which the
peers upgraded through the host's `ib0` fabric address. The run forced
`UCX_TLS=rc,ud,sm,self`, excluding UCP's TCP transport.

The ClosureHost reported `data plane: UCX`, the OpenFOAM client verified two
exact atomic field replacements through the resident ONNX model, both native
runners stopped cleanly, and the Slurm client completed. The driver reported
`Central OpenFOAM ONNX ClosureHost over UCX: PASS`. The CSC `argos-vars`
warning disabled only the auxiliary ARGOS integration, as in the earlier TCP
and UCX transport probes; it did not affect the allocation or data path.

The multi-rank central gate subsequently passed across an interactive host
allocation and a `small` client allocation. One resident ClosureHost accepted
global OpenFOAM ranks 0 and 1 from the client node. Each rank negotiated an
independent UCX channel, performed two exact atomic field replacements through
the shared ONNX model, and shut down its runner cleanly. The parallel OpenFOAM
probe and driver both reported `PASS`; arrival order was deliberately not
assumed, and rank 1 connected before rank 0 in this run.

The same two-rank topology passed under cross-allocation Longship supervision
with a `small` client job. Longship waited for the central host marker,
started the Slurm client proxy, observed successful client accounting, allowed
both UCX runners to consume shutdown, and removed the readiness marker. The
proxy and Longship both reported successful completion, so this run validates
the coupled success lifecycle as well as the field data path.

The split-allocation failure gates also passed. The client exited in
the intentional `FAILED` state; the Slurm proxy returned failure and Longship
terminated the waiting host and removed its marker. In the reverse direction,
an intentionally failed host caused Longship to terminate the proxy, whose
exit cleanup cancelled the client job; accounting reported `CANCELLED`.
No client allocation or readiness marker survived either test. Together these
runs validate fail-together propagation in both directions rather than only
the successful lifecycle.

### Solver-integrated LES acceptance

The first full solver gate passed across an interactive host allocation and a
`small` client allocation. Two OpenFOAM v2512 ranks ran on the client node.
The driver copied the source lid-driven-cavity LES case
into scratch, selected the runtime `nutFjord` model, and left the source case
unchanged.

Both solver ranks upgraded their independent Fjord sessions to UCX with
`UCX_TLS=rc,ud,sm,self`. The resident ClosureHost evaluated the native ONNX
artifact for every turbulence correction and published `nut` atomically. The
parallel `pimpleFoam` run completed times `0.001`, `0.002`, and `0.003`; both
native runners stopped, Slurm client accounting completed, Longship removed
its readiness marker, and the gate reported
`2-rank pimpleFoam nutFjord ONNX over central UCX: PASS`.

This run also exposed and then verified a transport-ordering fix: a UCX
upgrade now sends a one-byte UCP bootstrap before TCP readiness is
acknowledged. Endpoint establishment therefore cannot depend on the first
scientific tensor. The deterministic UCX regression checks that the server
finishes the upgrade before the client publishes any Rune payload.

OpenFOAM warned that LES is not strictly applicable to the two-dimensional
cavity. That warning is expected for this compact integration fixture and does
not weaken the software-path result; physical LES validation belongs to a
separate three-dimensional case campaign.

### One-equation LES acceptance

The compact `kEqnFjord` gate subsequently passed across an interactive host
allocation and a `small` client allocation. Two OpenFOAM v2512 client ranks selected
UCX with `UCX_TLS=rc,ud,sm,self`, and both native runners stopped cleanly.

The fixture was first loaded through ONNX Runtime and numerically verified for
the ordered 11-feature to 3-feature contract:

```text
k, grad(U), delta -> nut, kProduction, kDissipationCoeff
```

Parallel `pimpleFoam` selected `kEqnFjord`, solved the native k equation, wrote
all three closure fields, and completed times `0.001`, `0.002`, and `0.003`.
The Slurm proxy completed, Longship removed the readiness marker, and the gate
reported `2-rank pimpleFoam kEqnFjord ONNX over central UCX: PASS`.

This is a software-path acceptance result, not a turbulence-model validation.
The compact two-dimensional case used a seeded k field and an integration
fixture model. Its smooth solver reached the configured 1000-iteration ceiling
at each step while reducing the residual to approximately `3.6e-12`. A
three-dimensional NACA4412 campaign must retain its own k boundary conditions
and tune numerical controls before any physical or performance conclusions are
drawn.

The deferred NACA4412 software smoke also passed under the same split-allocation
pattern, with two OpenFOAM ranks in the `small` client job. Unlike the
compact cavity, this case supplied its own airfoil mesh, `0/k` field and wall
conditions, and PBiCG/DILU k solver. The copied case selected `kEqnFjord`, both
ranks used UCX, all three closure fields were written, three time steps
completed, both runners stopped, and Longship reported success. The source case
and its configured 16-way decomposition remained unchanged; the smoke copy was
decomposed into two ranks.

This NACA4412 result remains a software smoke, not physical validation or a
scalability benchmark. The integration driver used its conservative
`div(phi,k)` override, maximum Courant numbers reached approximately 63--79,
and the short pressure solves retained large final residuals. Non-fatal
function-object warnings also reflected requests for DES and a missing
processor-periodic patch in a non-DES two-rank smoke. A scientific campaign
must restore case-owned numerics, control the time step, establish convergence,
use a trained closure artifact, and compare against a declared baseline.

## Reproducing the site capability check

On another cluster, first load the same environment that will build and run
OpenFOAM, then inspect the effective stack rather than assuming that a module
name implies linkable development headers:

```bash
command -v ucx_info
command -v ucx_perftest
ucx_info -v

command -v ompi_info
ompi_info --parsable | grep -i ucx | head -40

module list 2>&1
```

For a future FoamNordic UCX build, also verify the UCP header and linker flags
through the site's supported package metadata or module variables:

```bash
pkg-config --modversion ucx 2>/dev/null || true
pkg-config --cflags --libs ucx 2>/dev/null || true
ucx_info -d
```

Avoid recursively searching an entire shared software tree: on large Spack
installations it is slow and may traverse many unrelated packages. If
`pkg-config` is unavailable, inspect the UCX prefix printed by `ucx_info -v`
and check only its `include/ucp/api/ucp.h` and `lib` directories.

## Portable acceptance matrix

Other HPC users should treat a transport as supported only after the matching
row passes on their site:

| Transport | Minimum acceptance test |
| --- | --- |
| UDS | Two native processes exchange and atomically commit Rune tensors over a named local endpoint |
| SHM | Separate processes negotiate UDS-to-SHM, wrap the ring, block/wake correctly, and clean up the named region |
| TCP | Server and client on different compute nodes verify payload, exchange identity, completion, and shutdown |
| UCX | FoamNordic is built with UCP, peers connect across compute nodes, the resolved plane reports UCX, and no TCP payload fallback occurs |

Loopback tests remain useful but do not replace a two-node TCP or UCX test.
Likewise, `ucx_perftest` is a site/network diagnostic and cannot replace the
FoamNordic Rune and Harbor correctness probe.
