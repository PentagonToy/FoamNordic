#!/usr/bin/env bash

set -Eeuo pipefail

transport=${FOAMNORDIC_NETWORK_TRANSPORT:-tcp}
case "$transport" in
    tcp) plane=TCP ;;
    ucx) plane=UCX ;;
    *) echo "[FoamNordic] Unknown network transport: $transport" >&2; exit 1 ;;
esac

fail()
{
    echo "[FoamNordic] Split-allocation $plane test failed: $*" >&2
    exit 1
}

for command in awk grep sacct sbatch scancel squeue; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
    || fail "start this driver inside the server's interactive allocation"

probe=${FOAMNORDIC_NETWORK_PROBE:?Set FOAMNORDIC_NETWORK_PROBE}
account=${FOAMNORDIC_SLURM_ACCOUNT:?Set FOAMNORDIC_SLURM_ACCOUNT}
partition=${FOAMNORDIC_NETWORK_CLIENT_PARTITION:-${FOAMNORDIC_TCP_CLIENT_PARTITION:-small}}
iterations=${FOAMNORDIC_NETWORK_ITERATIONS:-${FOAMNORDIC_TCP_ITERATIONS:-100}}
elements=${FOAMNORDIC_NETWORK_ELEMENTS:-${FOAMNORDIC_TCP_ELEMENTS:-131072}}
port=${FOAMNORDIC_CONTROL_PORT:-${FOAMNORDIC_TCP_PORT:-$((20000 + SLURM_JOB_ID % 20000))}}
client_wait=${FOAMNORDIC_NETWORK_CLIENT_WAIT:-${FOAMNORDIC_TCP_CLIENT_WAIT:-900}}
work_parent=${FOAMNORDIC_TEST_ROOT:?Set FOAMNORDIC_TEST_ROOT}

[[ -x "$probe" ]] || fail "network probe is not executable: $probe"
for value in "$iterations" "$elements" "$port" "$client_wait"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] \
        || fail "iterations, elements, port, and wait must be positive integers"
done
((port <= 65535)) || fail "control-plane port exceeds 65535"

server_node=$(hostname -s)
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

work_dir=$(mktemp -d "$work_parent/foamnordic-${transport}-split.XXXXXX")
server_log="$work_dir/server.log"
client_log="$work_dir/client-%j.log"
server_pid=
client_job=
completed=false

cleanup()
{
    if [[ "$completed" != true && -n "$client_job" ]]; then
        scancel "$client_job" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "[FoamNordic] $plane work directory: $work_dir"
echo "[FoamNordic] $plane server allocation: $SLURM_JOB_ID"
echo "[FoamNordic] $plane server node: $server_node"
echo "[FoamNordic] $plane client partition: $partition"
echo "[FoamNordic] TCP control address: $address"
if [[ "$transport" == ucx ]]; then
    echo "[FoamNordic] UCX interface address: $FOAMNORDIC_UCX_HOST"
fi

"$probe" server "$address" "$iterations" "$elements" "$transport" \
    >"$server_log" 2>&1 &
server_pid=$!

for _ in {1..100}; do
    grep -q "Inter-node server listening" "$server_log" 2>/dev/null && break
    kill -0 "$server_pid" 2>/dev/null \
        || fail "$plane server exited before readiness"
    sleep 0.1
done
grep -q "Inter-node server listening" "$server_log" \
    || fail "$plane server did not become ready"

printf -v client_command \
    'exec %q client %q %q %q %q' \
    "$probe" "$address" "$iterations" "$elements" "$transport"

client_job=$(
    sbatch \
        --parsable \
        --account="$account" \
        --partition="$partition" \
        --time=00:10:00 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --exclude="$server_node" \
        --job-name="fn-${transport}-client" \
        --output="$client_log" \
        --export=ALL \
        --wrap="$client_command"
)
client_log=${client_log/\%j/$client_job}

echo "[FoamNordic] $plane client job: $client_job"
echo "[FoamNordic] $plane client log: $client_log"

client_deadline=$((SECONDS + client_wait))
while kill -0 "$server_pid" 2>/dev/null; do
    state=$(
        sacct --noheader --allocations --jobs="$client_job" --format=State \
            | awk 'NF {print $1; exit}'
    )
    case "$state" in
        FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*)
            fail "$plane client allocation ended in state $state"
            ;;
    esac
    ((SECONDS < client_deadline)) \
        || fail "$plane client did not complete within ${client_wait}s"
    sleep 1
done

set +e
wait "$server_pid"
server_status=$?
set -e
server_pid=
[[ "$server_status" -eq 0 ]] \
    || fail "$plane server exited with status $server_status"

while squeue --noheader --jobs="$client_job" | grep -q .; do
    sleep 1
done

client_state=$(
    sacct --noheader --allocations --jobs="$client_job" --format=State \
        | awk 'NF {print $1; exit}'
)
[[ "$client_state" == COMPLETED* ]] \
    || fail "$plane client allocation ended in state ${client_state:-unknown}"

grep -q "Inter-node server: PASS" "$server_log" \
    || fail "$plane server did not report success"
grep -q "Inter-node client: PASS" "$client_log" \
    || fail "$plane client did not report success"
grep -q "Data plane: $plane" "$client_log" \
    || fail "split-allocation probe did not retain $plane"

completed=true
sed -n '1,140p' "$server_log"
sed -n '1,180p' "$client_log"
echo "[FoamNordic] Split-allocation Fjord $plane: PASS"
