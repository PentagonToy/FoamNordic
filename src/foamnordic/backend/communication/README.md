# Fjord communication core

Fjord is FoamNordic's native data path. It has no external database or message
broker dependency.

The names describe three deliberately small responsibilities:

- **Rune** defines the versioned binary tensor frame.
- **Fjord** moves an exact number of bytes through a channel.
- **Harbor** sends and receives tensors over a Fjord channel.

The current channels support an in-process Unix socket pair, a named Unix
socket for local processes, and TCP for processes on different nodes.
Shared-memory and UCX channels must implement the same `FjordChannel`
interface without changing Rune or Harbor.

Endpoints use explicit FoamNordic addresses:

```text
unix:///tmp/foamnordic.sock
tcp://127.0.0.1:2026
```

Unix sockets are intended for a workstation or one compute node. TCP is the
portable baseline between Slurm nodes. A later runtime policy may choose the
channel, but the communication core never guesses or silently changes it.

## Rune version 1

Every integer is encoded little-endian. A frame contains:

1. a fixed 72-byte prefix;
2. a UTF-8 tensor name;
3. one unsigned 64-bit extent per dimension;
4. the contiguous tensor payload.

The prefix records the protocol version, message kind, capabilities, element
type, dimension count, metadata lengths, payload length, exchange index, MPI
rank identity, physical time, session ID, and negotiated payload limit. Rune
does not prescribe an OpenFOAM field, Python object, transport, scheduler, or
model backend.

Every connection begins with `hello` and `hello_accept` or `hello_reject`.
Tensor, completion, shutdown, and error messages then use the selected session
ID. Capability selection is an intersection: a peer cannot activate UDS, SHM,
TCP, or UCX unless both sides advertised it.

## Native verification

```bash
cmake --log-level=WARNING \
    -S . \
    -B build/fjord \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFOAMNORDIC_TESTS=ON \
    -DFOAMNORDIC_BENCHMARKS=ON

cmake --build build/fjord --parallel -- --quiet
ctest --test-dir build/fjord --output-on-failure
build/fjord/tools/benchmarks/foamnordic_fjord_benchmark
```
