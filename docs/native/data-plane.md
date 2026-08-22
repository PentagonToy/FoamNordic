# FoamNordic native data plane

FoamNordic separates orchestration from field movement. Python may configure
an experiment, but it must not copy, concatenate, poll, or serialize every
OpenFOAM field during each exchange.

The native data plane has three permanent layers:

```text
OpenFOAM / Python / native model
              │
           Harbor        tensor and exchange semantics
              │
            Rune         versioned binary representation
              │
            Fjord        exact-byte channel contract
              │
     UDS / SHM / TCP / UCX
```

No transport may define its own tensor format or exchange lifecycle.

## Permanent contracts

### Rune

Rune owns byte order, scalar type, shape, field name, exchange index, solver
time index, and physical time. A Rune frame has the same representation over
every channel. Rune v2 keeps the monotonic exchange index separate from the
solver time index, so repeated PIMPLE/PISO correctors at one OpenFOAM time are
distinct atomic calls without losing their shared solver-time identity.
Protocol changes require a version change and compatibility tests.

Rune includes a small session handshake and message kind. These fields support
capability negotiation, MPI rank identity, completion, and clean shutdown
without transport-specific messages. Multiple tensor messages followed by one
completion message form an ordered exchange batch without introducing a
transport-specific envelope.

### Fjord

Every Fjord channel implements only these operations:

```text
write_all(bytes)
read_all(bytes)
close()
```

The contract is blocking and exact. Partial operating-system reads and writes
are completed inside the channel. Timeouts and cancellation will be common
channel options rather than backend-specific loops.

### Harbor

Harbor translates Rune frames to and from an ordered message stream containing
tensors, completion markers, shutdown, and errors. It owns safety limits and
session validation. A native closure runner consumes this stream and gives
`complete` its batch-boundary meaning. Harbor itself does not choose a
transport and does not know about Slurm, OpenFOAM dictionaries, JAX, ONNX, or
Python objects.

Harbor defaults to a blocking handshake. This is intentional for LES closure
coupling: an OpenFOAM exchange must not advance with an unconfirmed peer or a
partially negotiated data path. `Experiment.start(block=False)` may launch a
workload asynchronously, but it does not weaken the field-exchange handshake.

Native handshake policies are:

- `blocking`: wait until the peer accepts or rejects the session;
- `timed`: preserve blocking semantics with an explicit failure deadline;
- `disabled`: skip negotiation only when both peers were configured out of
  band with the same session contract.

The future Python API controls this policy, while the native default remains
`blocking`.

## Channel roles

### Unix-domain socket (UDS)

UDS is the local-process baseline on macOS and Linux. It is also the local
control channel used to negotiate shared-memory regions. It carries complete
Rune frames when shared memory is unavailable or unnecessary.

### POSIX shared memory (SHM)

SHM is the same-node bulk channel. It will use session-scoped regions and two
single-producer/single-consumer rings, one in each direction. Atomic sequence
counters publish complete slots; a reader must never observe a partially
written tensor.

The native `SharedSlotRing` implements the publication primitive. Every fixed
slot owns a monotonically increasing sequence counter. A producer writes size
and payload before a release publication; a consumer acquires that sequence
before reading and releases the slot only after copying the complete message.
Unread slots are never overwritten, and wraparound does not reset sequence
identity. The ring requires lock-free 64-bit atomics and rejects unsupported
platforms at build time.

The ring is deliberately separate from POSIX object naming and waiting. The
`NamedShmChannel` now maps two rings into one session-scoped `shm_open` region.
The creator initializes a versioned region header and owns `shm_unlink`; a peer
opens the same name independently, validates the declared layout, and selects
the opposite transmit/receive rings. macOS may report a page-rounded mapping
size, so the declared region must be completely contained in the mapped object
rather than byte-for-byte equal to its reported size.

A cross-process test reconnects after `fork`, negotiates a Harbor SHM session,
and round-trips a Rune tensor. The wakeup layer keeps UDS open after rendezvous
and blocks only when a ring changes between empty and non-empty or full and
writable; the data itself remains in SHM.

UDS performs setup, capability exchange, descriptor/name transfer, error
reporting, and shutdown. SHM carries only negotiated Rune frames. Linux x86-64
and Linux AArch64 use process-shared futex wait/wake operations on the region
header. macOS uses the already-connected UDS as its blocking event primitive.
This is intentionally hidden behind the channel contract: `std::atomic::wait`
is not used across processes because the C++ standard does not guarantee that
an implementation will wake an atomic object in a separately mapped process.
Busy polling is not part of the production design.

The support order is deliberate:

1. Linux x86-64: primary HPC data plane, SHM plus futex;
2. macOS arm64: development data plane, SHM plus UDS wakeup;
3. Linux AArch64: portable HPC data plane, the same SHM plus futex contract.

The UDS-to-SHM upgrade itself is blocking and does not poll for a filesystem
object. After the initial Rune capability handshake, the accepting peer sends
one session-scoped SHM name. The connecting peer duplicates its established
UDS endpoint, creates the two-ring region with that duplicate reserved for
wakeups, and returns one `shm_ready` record over the original UDS descriptor.
Only then does the accepting peer map the region and replace its Harbor data
channel. The original descriptor is closed after the transition; the duplicate
stays attached to SHM solely as the macOS wake/control path. Linux uses the same
handshake but waits on futex words in the shared region after the upgrade.

### Atomic exchange publication

FoamNordic preserves the useful semantic guarantee historically provided by
an atomic Lua commit without retaining Redis or Lua. A producer prepares all
Rune tensor frames for one monotonically increasing exchange and then appends
one `complete` record containing the exchange index and exact tensor count.
Release publication in the SPSC ring preserves this order. The native closure
runner may collect prepared tensors, but it cannot validate, scale, infer, or
modify an OpenFOAM field until it acquires and validates that commit record.
An incomplete batch, mismatched index, or mismatched tensor count fails the
exchange rather than exposing partial state.

`AtomicFieldExchange` is the transport-independent boundary used by the future
OpenFOAM adapter. It publishes every configured input field as one batch and
buffers returned tensors in native memory. Output sizes, types, shapes, names,
exchange identity, and the final tensor count are all validated before the
first byte of an OpenFOAM-owned destination is modified. A failed or incomplete
response therefore leaves all solver fields unchanged; there is no partially
updated time step for the solver to advance from.

### TCP

TCP is the portable inter-node baseline and must work on a workstation,
Ethernet cluster, and Slurm allocation without optional libraries. Each socket
uses exact Rune framing and `TCP_NODELAY`. The runtime resolves an explicitly
selected network interface to an address before starting a listener.

### UCX

UCX is an optional inter-node bulk channel, not a required runtime dependency.
TCP performs rendezvous and capability negotiation. UCX is selected only when
both peers advertise a compatible build and a connectivity probe succeeds.

The optional `UcxChannel` uses the UCP stream API and preserves Fjord's exact
byte contract. A TCP Harbor first negotiates `tcp|ucx`, the server advertises a
UCP listener address, and the client sends a one-byte bootstrap over UCP before
acknowledging readiness on TCP. This transport bootstrap progresses and proves
the endpoint independently of the first scientific payload. Both peers then
replace their payload channel before the first Rune tensor is sent. TCP remains
the rendezvous/control path; Rune tensor and completion frames use UCP after a
successful upgrade. This is a direct FoamNordic UCP integration and does not
route field traffic through MPI's UCX PML.

Automatic mode may fall back from UCX to TCP during negotiation. An explicit
`ucx` request must fail clearly rather than silently use TCP. A live session
never changes channels halfway through an exchange.

UCX is disabled by default so a workstation and an HPC site without UCX retain
the same build. Enable it with `FOAMNORDIC_UCX=ON`; an installation outside the
compiler's normal search path can be supplied through `FOAMNORDIC_UCX_ROOT`.

## Placement policy

The runtime owns selection; channel classes do not inspect Slurm variables or
guess topology.

```text
same process                          direct view (future)
same node, SHM negotiated             SHM with UDS control
same node, no SHM                     UDS
different node, UCX negotiated        UCX with TCP control
different node, no UCX                TCP
```

Users may override automatic selection. The resolved channel and reason must
be included in the experiment summary.

## MPI layout

OpenFOAM ranks do not gather all fields through rank zero. Each rank keeps its
own field view and exchange identity. The client may present the rank tensors
as one logical field, but aggregation is optional and must not be required by
the wire protocol.

This avoids a rank-zero memory spike and preserves a path toward one channel
per rank, one channel per node, or UCX multi-rail operation.

## Buffer ownership

- OpenFOAM owns outbound field memory until the channel reports publication.
- A channel never retains a caller's `TensorView` after `send` returns.
- Received `Tensor` storage is owned by Harbor until moved to its consumer.
- SHM slots return to the producer only after the consumer advances its
  sequence counter.
- Python bindings expose NumPy views only while the native owner remains alive.

## Test layout

Tests live outside production sources:

```text
tests/
├── backend/
│   ├── communication/
│   └── inference/
├── client/
├── openfoam/
└── runtime/
```

Unit tests require no OpenFOAM or Slurm. Integration tests state their external
requirements explicitly. Benchmarks live under `tools/benchmarks` and are not
treated as correctness tests.

The Fjord benchmark models a small LES combustion exchange rather than a raw
single-buffer copy: three 400-cell float64 input fields are prepared and
atomically committed, then one 400-cell output field is returned and committed.
It reports complete exchanges per second and useful field-payload throughput
for UDS, same-process SHM, and named cross-process SHM. The last path exercises
the production wait primitive: futex on Linux and UDS wakeup on macOS. This
microbenchmark detects transport regressions, but the release performance gate
remains an a-posteriori OpenFOAM case: coupled wall time must remain at or below
1.5 times the uncoupled solver wall time under the same allocation,
decomposition, time-step sequence, and output policy.

The transport matrix covers Rune codec, socket pair, named UDS, loopback TCP,
SHM wraparound and crash recovery, two-node TCP, and two-node UCX. A
UCX-enabled build also runs a deterministic TCP-to-UCX loopback upgrade. A
Slurm HPC site passed the fabric-backed split-allocation probe with UCX's TCP transport
explicitly excluded. A central multi-node OpenFOAM and ClosureHost run remains
a higher-level integration gate rather than a Fjord transport gate.

A reproducible two-node TCP driver is available at
`tools/network/testTcpSlurm.sh`. It requires an executable
`foamnordic_fjord_network_probe` built with
`FOAMNORDIC_NETWORK_TOOLS=ON` and an allocation containing at least two nodes.

Site partition limits matter independently of the number of nodes shown by
`sinfo`: `small` and `interactive` accept one node per job, `test` accepts one
or two nodes but may not have two nodes simultaneously available, and
multi-node `medium` reserves complete nodes. When a two-node allocation is
impractical, `tools/network/testTcpSplitSlurm.sh` runs the server inside an
existing one-node interactive allocation and submits a one-node `small` client
job with the server node explicitly excluded. The shared probe executable and
logs remain on project scratch; only TCP payloads cross the node boundary.

The matching UCX entry points are `tools/network/testUcxSlurm.sh` and
`tools/network/testUcxSplitSlurm.sh`. TCP and UCX wrappers share scheduler and
accounting logic, while the UCX wrapper advertises the IPv4 address of `ib0` by
default. Sites with a different fabric name set `FOAMNORDIC_UCX_INTERFACE`, or
provide the exact listener address through `FOAMNORDIC_UCX_HOST`.

`foamnordic_fjord_network_probe` is the two-node TCP and UCX correctness probe.
It publishes one tensor plus one atomic completion record per closure call,
verifies monotonic exchange and solver-time indices on both peers, checks the
returned payload, and reports round-trip and payload rates. It intentionally
contains no Slurm API: the scheduler only places its server and client
processes on different nodes.

The current implementation and site-validation matrix, including an example
CSC Roihu UCX 1.20.0 environment and the distinction between an available runtime and an
implemented Fjord channel, is recorded in
[HPC transport status and site notes](hpc-transport-status.md).
