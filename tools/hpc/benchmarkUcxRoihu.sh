#!/usr/bin/env bash
#SBATCH --job-name=fno-ucx
#SBATCH --account=project_2015384
#SBATCH --partition=medium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2015384/Hanseul/Tests/foamnordic-ucx-%j.out
#SBATCH --error=/scratch/project_2015384/Hanseul/Tests/foamnordic-ucx-%j.err

set -Eeuo pipefail

module load openfoam/2512

source_dir=/scratch/project_2015384/Hanseul/Source/FoamNordic
work_root=/scratch/project_2015384/Hanseul/Tests
build_dir=$work_root/.foamnordic-ucx-build-${SLURM_JOB_ID}
run_dir=$work_root/.foamnordic-ucx-run-${SLURM_JOB_ID}
ucx_root=$(pkg-config --variable=prefix ucx)

export UCX_TLS=rc,ud,sm,self
export FOAMNORDIC_UCX_INTERFACE=${FOAMNORDIC_UCX_INTERFACE:-ib0}

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} -eq 2 ]] || {
    echo "Exactly two nodes are required" >&2
    exit 2
}
server_node=${nodes[0]}
client_node=${nodes[1]}
server_pid=

cleanup()
{
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    find "$build_dir" "$run_dir" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$build_dir" "$run_dir" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$build_dir" "$run_dir"

export FOAMNORDIC_UCX_HOST
FOAMNORDIC_UCX_HOST=$(
    srun --nodes=1 --ntasks=1 --nodelist="$server_node" \
        ip -o -4 address show dev "$FOAMNORDIC_UCX_INTERFACE" \
        | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}'
)
[[ -n "$FOAMNORDIC_UCX_HOST" ]] || {
    echo "The UCX interface has no IPv4 address" >&2
    exit 2
}

echo "FoamNordic commit: $(git -C "$source_dir" rev-parse HEAD)"
echo "UCX: $(pkg-config --modversion ucx)"
echo "UCX root: $ucx_root"
echo "UCX transports: $UCX_TLS"
echo "UCX interface: $FOAMNORDIC_UCX_INTERFACE ($FOAMNORDIC_UCX_HOST)"
echo "Server node: $server_node"
echo "Client node: $client_node"

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFOAMNORDIC_TESTS=OFF \
    -DFOAMNORDIC_NETWORK_TOOLS=ON \
    -DFOAMNORDIC_UCX=ON \
    -DFOAMNORDIC_UCX_ROOT="$ucx_root"
cmake --build "$build_dir" \
    --target foamnordic_fjord_network_probe \
    --parallel "${SLURM_CPUS_PER_TASK:-1}"

probe=$build_dir/tools/network/foamnordic_fjord_network_probe
[[ -x "$probe" ]] || {
    echo "Network probe was not built" >&2
    exit 2
}

run_probe()
{
    local label=$1
    local repetition=$2
    local iterations=$3
    local elements=$4
    local port=$((20000 + SLURM_JOB_ID % 20000 + repetition))
    local address=tcp://$server_node:$port
    local server_log=$run_dir/${label}-${repetition}-server.log
    local client_log=$run_dir/${label}-${repetition}-client.log

    srun --exclusive --nodes=1 --ntasks=1 --nodelist="$server_node" \
        "$probe" server "$address" "$iterations" "$elements" ucx \
        >"$server_log" 2>&1 &
    server_pid=$!

    for _ in {1..200}; do
        grep -q "Inter-node server listening" "$server_log" 2>/dev/null && break
        kill -0 "$server_pid" 2>/dev/null || {
            sed -n '1,160p' "$server_log"
            return 1
        }
        sleep 0.05
    done
    grep -q "Inter-node server listening" "$server_log" || {
        echo "Server readiness timed out" >&2
        return 1
    }

    srun --exclusive --nodes=1 --ntasks=1 --nodelist="$client_node" \
        "$probe" client "$address" "$iterations" "$elements" ucx \
        >"$client_log" 2>&1
    wait "$server_pid"
    server_pid=

    grep -q "Inter-node server: PASS" "$server_log"
    grep -q "Inter-node client: PASS" "$client_log"
    grep -q "Data plane: UCX" "$client_log"
    echo "BENCHMARK $label RUN $repetition"
    sed -n '1,120p' "$client_log"
}

for repetition in 1 2 3 4 5 6 7; do
    run_probe small "$repetition" 100000 400
done

for repetition in 1 2 3 4 5 6 7; do
    run_probe large "$((repetition + 10))" 100 131072
done

echo "FoamNordic two-node UCX benchmark: PASS"
