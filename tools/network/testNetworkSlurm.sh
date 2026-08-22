#!/usr/bin/env bash

set -Eeuo pipefail

transport=${FOAMNORDIC_NETWORK_TRANSPORT:-tcp}
case "$transport" in
    tcp) plane=TCP ;;
    ucx) plane=UCX ;;
    *) echo "[FoamNord] Unknown network transport: $transport" >&2; exit 1 ;;
esac

fail()
{
    echo "[FoamNord] Inter-node $plane test failed: $*" >&2
    exit 1
}

for command in awk grep scontrol srun; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] \
    || fail "run this test inside a Slurm allocation"

probe=${FOAMNORDIC_NETWORK_PROBE:?Set FOAMNORDIC_NETWORK_PROBE}
[[ -x "$probe" ]] || fail "network probe is not executable: $probe"

iterations=${FOAMNORDIC_NETWORK_ITERATIONS:-${FOAMNORDIC_TCP_ITERATIONS:-100}}
elements=${FOAMNORDIC_NETWORK_ELEMENTS:-${FOAMNORDIC_TCP_ELEMENTS:-131072}}
port=${FOAMNORDIC_CONTROL_PORT:-${FOAMNORDIC_TCP_PORT:-$((20000 + SLURM_JOB_ID % 20000))}}
for value in "$iterations" "$elements" "$port"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] \
        || fail "iterations, elements, and port must be positive integers"
done
((port <= 65535)) || fail "control-plane port exceeds 65535"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
((${#nodes[@]} >= 2)) || fail "inter-node $plane requires at least two nodes"

server_node=${nodes[0]}
client_node=${nodes[1]}
address="tcp://$server_node:$port"
if [[ "$transport" == ucx && -z "${FOAMNORDIC_UCX_HOST:-}" ]]; then
    command -v ip >/dev/null || fail "ip is required to resolve the UCX interface"
    ucx_interface=${FOAMNORDIC_UCX_INTERFACE:-ib0}
    FOAMNORDIC_UCX_HOST=$(
        ip -o -4 address show dev "$ucx_interface" \
            | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}'
    )
    [[ -n "$FOAMNORDIC_UCX_HOST" ]] \
        || fail "interface $ucx_interface has no IPv4 address"
    export FOAMNORDIC_UCX_HOST
fi

work_parent=${FOAMNORDIC_TEST_ROOT:-${TMPDIR:-/tmp}}
work_dir=$(mktemp -d "$work_parent/foamnordic-${transport}.XXXXXX")
server_log="$work_dir/server.log"
client_log="$work_dir/client.log"
server_step=

cleanup()
{
    if [[ -n "$server_step" ]]; then
        kill "$server_step" 2>/dev/null || true
        wait "$server_step" 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "[FoamNord] $plane work directory: $work_dir"
echo "[FoamNord] $plane server node: $server_node"
echo "[FoamNord] $plane client node: $client_node"
echo "[FoamNord] TCP control address: $address"
if [[ "$transport" == ucx ]]; then
    echo "[FoamNord] UCX interface address: $FOAMNORDIC_UCX_HOST"
fi

srun --nodes=1 --ntasks=1 --nodelist="$server_node" --exact --exclusive \
    "$probe" server "$address" "$iterations" "$elements" "$transport" \
    >"$server_log" 2>&1 &
server_step=$!

for _ in {1..100}; do
    grep -q "Inter-node server listening" "$server_log" 2>/dev/null && break
    kill -0 "$server_step" 2>/dev/null \
        || fail "$plane server step exited before readiness"
    sleep 0.1
done
grep -q "Inter-node server listening" "$server_log" \
    || fail "$plane server did not become ready"

srun --nodes=1 --ntasks=1 --nodelist="$client_node" --exact --exclusive \
    "$probe" client "$address" "$iterations" "$elements" "$transport" \
    >"$client_log" 2>&1

wait "$server_step"
server_step=

grep -q "Inter-node server: PASS" "$server_log" \
    || fail "$plane server did not report success"
grep -q "Inter-node client: PASS" "$client_log" \
    || fail "$plane client did not report success"
grep -q "Data plane: $plane" "$client_log" \
    || fail "inter-node probe did not retain $plane"

sed -n '1,140p' "$server_log"
sed -n '1,180p' "$client_log"
echo "[FoamNord] Two-node Fjord $plane: PASS"
