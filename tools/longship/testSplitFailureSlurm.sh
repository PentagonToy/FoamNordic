#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNord] Split Longship failure test failed: $*" >&2
    exit 1
}

for command in awk grep sacct scancel squeue; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done
[[ -n "${SLURM_JOB_ID:-}" ]] \
    || fail "run this test inside an interactive host allocation"

repository=$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)
longship=${FOAMNORDIC_LONGSHIP:?Set FOAMNORDIC_LONGSHIP}
account=${FOAMNORDIC_SLURM_ACCOUNT:?Set FOAMNORDIC_SLURM_ACCOUNT}
partition=${FOAMNORDIC_LONGSHIP_CLIENT_PARTITION:-small}
work_parent=${FOAMNORDIC_TEST_ROOT:?Set FOAMNORDIC_TEST_ROOT}
wait_seconds=${FOAMNORDIC_LONGSHIP_FAILURE_WAIT:-300}
[[ -x "$longship" ]] || fail "Longship executable is unavailable: $longship"
[[ "$wait_seconds" =~ ^[1-9][0-9]*$ ]] \
    || fail "FOAMNORDIC_LONGSHIP_FAILURE_WAIT must be positive"

server_node=$(hostname -s)
work_dir=$(mktemp -d "$work_parent/foamnordic-longship-failure.XXXXXX")
client_jobs=()

cleanup()
{
    local job
    for job in "${client_jobs[@]}"; do
        scancel "$job" 2>/dev/null || true
    done
}
trap cleanup EXIT

wait_for_job_id()
{
    local file=$1
    for _ in {1..100}; do
        [[ -s "$file" ]] && return 0
        sleep 0.1
    done
    fail "Slurm proxy did not record a client job in $file"
}

accounting_state()
{
    sacct --noheader --allocations --jobs="$1" --format=State \
        | awk 'NF {print $1; exit}'
}

wait_for_terminal_state()
{
    local job=$1
    local deadline=$((SECONDS + wait_seconds))
    local state=
    while ((SECONDS < deadline)); do
        state=$(accounting_state "$job")
        case "$state" in
            COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*)
                printf '%s\n' "$state"
                return 0
                ;;
        esac
        sleep 1
    done
    fail "job $job did not reach a terminal state"
}

run_client_failure_gate()
{
    local prefix="$work_dir/client-failure"
    local ready="$prefix.ready"
    local job_file="$prefix.job"
    local status

    set +e
    "$longship" \
        --ready "$ready" \
        --host-output "$prefix-host.log" \
        --solver-output "$prefix-proxy.log" \
        --readiness-timeout-ms 30000 \
        --termination-grace-ms 5000 \
        --host \
        /bin/sh -c 'touch "$1"; exec sleep 300' sh "$ready" \
        --solver \
        "$repository/tools/longship/runSlurmClient.sh" \
        --account "$account" \
        --partition "$partition" \
        --time 00:05:00 \
        --nodes 1 \
        --ntasks 1 \
        --cpus-per-task 1 \
        --exclude "$server_node" \
        --job-name fn-client-fail \
        --output "$prefix-client-%j.log" \
        --job-id-file "$job_file" \
        --wait-seconds "$wait_seconds" \
        -- \
        /bin/sh -c 'exit 23' \
        >"$prefix-longship.log" 2>&1
    status=$?
    set -e

    [[ "$status" -ne 0 ]] \
        || fail "Longship accepted an intentionally failed client"
    wait_for_job_id "$job_file"
    local job
    job=$(<"$job_file")
    client_jobs+=("$job")
    local state
    state=$(wait_for_terminal_state "$job")
    [[ "$state" == FAILED* ]] \
        || fail "intentional client failure ended in state $state"
    [[ ! -e "$ready" ]] \
        || fail "client failure left the host readiness marker"
    grep -q "Solver exited with a failure status" "$prefix-longship.log" \
        || fail "Longship did not attribute the client failure"

    echo "[FoamNord] Client failure job: $job ($state)"
    echo "[FoamNord] Client failure terminated ClosureHost: PASS"
}

run_host_failure_gate()
{
    local prefix="$work_dir/host-failure"
    local ready="$prefix.ready"
    local job_file="$prefix.job"
    local status

    set +e
    "$longship" \
        --ready "$ready" \
        --host-output "$prefix-host.log" \
        --solver-output "$prefix-proxy.log" \
        --readiness-timeout-ms 30000 \
        --termination-grace-ms 5000 \
        --host \
        /bin/sh -c 'touch "$1"; sleep 10; exit 17' sh "$ready" \
        --solver \
        "$repository/tools/longship/runSlurmClient.sh" \
        --account "$account" \
        --partition "$partition" \
        --time 00:05:00 \
        --nodes 1 \
        --ntasks 1 \
        --cpus-per-task 1 \
        --exclude "$server_node" \
        --job-name fn-host-fail \
        --output "$prefix-client-%j.log" \
        --job-id-file "$job_file" \
        --wait-seconds "$wait_seconds" \
        -- \
        /bin/sh -c 'sleep 300' \
        >"$prefix-longship.log" 2>&1
    status=$?
    set -e

    [[ "$status" -ne 0 ]] \
        || fail "Longship accepted an intentionally failed host"
    wait_for_job_id "$job_file"
    local job
    job=$(<"$job_file")
    client_jobs+=("$job")
    local state
    state=$(wait_for_terminal_state "$job")
    [[ "$state" == CANCELLED* ]] \
        || fail "host failure left client job $job in state $state"
    [[ ! -e "$ready" ]] \
        || fail "host failure left its readiness marker"
    grep -q "ClosureHost exited before the solver completed" \
        "$prefix-longship.log" \
        || fail "Longship did not attribute the host failure"

    echo "[FoamNord] Host failure client job: $job ($state)"
    echo "[FoamNord] Host failure cancelled Slurm client: PASS"
}

echo "[FoamNord] Split Longship failure work directory: $work_dir"
run_client_failure_gate
run_host_failure_gate
client_jobs=()
echo "[FoamNord] Split-allocation Longship fail-together lifecycle: PASS"
