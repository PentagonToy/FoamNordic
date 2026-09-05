#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNordic] Closure hook test failed: $*" >&2
    exit 1
}

for command in cmake c++ ctest find foamDictionary shasum wclean wmake; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done

[[ -n "${WM_PROJECT_VERSION:-}" ]] \
    || fail "load an OpenFOAM environment before running this test"

repository=$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)
source_case=${FOAMNORDIC_SOURCE_CASE:?Set FOAMNORDIC_SOURCE_CASE}
ort_root=${FOAMNORDIC_ONNX_RUNTIME_ROOT:?Set FOAMNORDIC_ONNX_RUNTIME_ROOT}
[[ -d "$source_case/system" ]] \
    || fail "FOAMNORDIC_SOURCE_CASE is not an OpenFOAM case"
[[ -f "$ort_root/VERSION_NUMBER" ]] \
    || fail "FOAMNORDIC_ONNX_RUNTIME_ROOT is not an ONNX Runtime installation"
jobs=${FOAMNORDIC_BUILD_JOBS:-2}
mpi_ranks=${FOAMNORDIC_MPI_RANKS:-1}
test_timeout=${FOAMNORDIC_TEST_TIMEOUT:-60}
[[ "$test_timeout" =~ ^[1-9][0-9]*$ ]] \
    || fail "FOAMNORDIC_TEST_TIMEOUT must be a positive integer"
[[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]] \
    || fail "FOAMNORDIC_MPI_RANKS must be a positive integer"
work_parent=${FOAMNORDIC_TEST_ROOT:-${TMPDIR:-/tmp}}
resume_work_dir=${FOAMNORDIC_RESUME_UCX_WORK_DIR:-}
if [[ -n "$resume_work_dir" ]]; then
    work_dir=$resume_work_dir
else
    work_dir=$(mktemp -d "$work_parent/foamnordic-onnx-hook.XXXXXX")
fi
build_dir="$work_dir/build"
case_dir="$work_dir/case"
model_dir="$work_dir/model"
address="unix://$work_dir/closure.sock"
closure_shared_memory=true
closure_ucx=false
worker_pid=
probe_pid=
watchdog_pid=
worker_pids=()
split_lock_fd=

cleanup()
{
    if [[ -n "$worker_pid" ]]; then
        kill "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$probe_pid" ]]; then
        kill "$probe_pid" 2>/dev/null || true
        wait "$probe_pid" 2>/dev/null || true
    fi
    if [[ -n "$watchdog_pid" ]]; then
        kill "$watchdog_pid" 2>/dev/null || true
        wait "$watchdog_pid" 2>/dev/null || true
    fi
    for pid in "${worker_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT

configure_probe()
{
    local key=$1
    local expression=$2
    local output=$3
    local scale=$4
    local seed=$5
    local expect_failure=$6
    local patch=${7:-none}

    cmake \
        -DINPUT="$repository/tools/template/openfoam/closureDict.in" \
        -DOUTPUT="$case_dir/system/foamnordicClosureDict" \
        -DFOAMNORDIC_ADDRESS="$address" \
        -DFOAMNORDIC_SESSION_ID=1 \
        -DFOAMNORDIC_SHARED_MEMORY="$closure_shared_memory" \
        -DFOAMNORDIC_UCX="$closure_ucx" \
        -DFOAMNORDIC_INPUT_KEYS="$key" \
        -DFOAMNORDIC_INPUT_EXPRESSIONS="\"$expression\"" \
        -DFOAMNORDIC_INPUT_PATCHES="$patch" \
        -DFOAMNORDIC_OUTPUT_FIELDS="$output" \
        -DFOAMNORDIC_OUTPUT_PATCHES="$patch" \
        -DFOAMNORDIC_PROBE_EXPRESSION="$expression" \
        -DFOAMNORDIC_PROBE_OUTPUT="$output" \
        -DFOAMNORDIC_PROBE_PATCH="$patch" \
        -DFOAMNORDIC_PROBE_SCALE="$scale" \
        -DFOAMNORDIC_PROBE_SEED="$seed" \
        -DFOAMNORDIC_PROBE_EXPECT_FAILURE="$expect_failure" \
        -P "$repository/tools/openfoam/configureClosureDict.cmake"
}

run_rejection_probe()
{
    local log=$1

    "$build_dir/tools/openfoam/foamnordic_openfoam_echo" \
        "$address" \
        U \
        1.0 \
        --reject \
        >"$log" 2>&1 &
    worker_pid=$!
    wait_for_endpoint
    run_probe
    wait_bounded "$worker_pid" "rejecting native echo worker"
    worker_pid=
    [[ ! -e "$work_dir/closure.sock" ]] \
        || fail "rejecting echo worker left a stale UDS endpoint"
    grep -q "Rejected closure call 0" "$log" \
        || fail "native echo worker did not reject the active exchange"
    sed -n '1,100p' "$log"
}

wait_for_endpoint()
{
    for _ in {1..50}; do
        [[ -S "$work_dir/closure.sock" ]] && return
        sleep 0.1
    done
    fail "native worker did not create its UDS endpoint"
}

wait_for_rank_endpoints()
{
    local rank
    local ready
    for _ in {1..100}; do
        ready=true
        for ((rank = 0; rank < mpi_ranks; ++rank)); do
            [[ -S "$work_dir/closure-$rank.sock" ]] || ready=false
        done
        [[ "$ready" = true ]] && return
        sleep 0.1
    done
    fail "rank-local workers did not create every UDS endpoint"
}

wait_bounded()
{
    local pid=$1
    local label=$2
    local status

    (
        sleep "$test_timeout"
        kill -TERM "$pid" 2>/dev/null || true
    ) &
    watchdog_pid=$!

    set +e
    wait "$pid"
    status=$?
    set -e

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    watchdog_pid=

    if [[ "$status" -eq 143 ]]; then
        fail "$label exceeded ${test_timeout}s"
    fi
    [[ "$status" -eq 0 ]] \
        || fail "$label exited with status $status"
}

run_probe()
{
    "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
        -case "$case_dir" &
    probe_pid=$!
    wait_bounded "$probe_pid" "OpenFOAM closure probe"
    probe_pid=
}

run_echo_probe()
{
    local output=$1
    local scale=$2
    local log=$3

    "$build_dir/tools/openfoam/foamnordic_openfoam_echo" \
        "$address" \
        "$output" \
        "$scale" \
        >"$log" 2>&1 &
    worker_pid=$!
    wait_for_endpoint
    run_probe
    wait_bounded "$worker_pid" "native echo worker"
    worker_pid=
    [[ ! -e "$work_dir/closure.sock" ]] \
        || fail "native echo worker left a stale UDS endpoint"
    grep -q "Data plane: SHM" "$log" \
        || fail "native echo worker did not negotiate SHM"
    [[ $(grep -c "Same-time closure call" "$log") -eq 2 ]] \
        || fail "native echo worker did not observe two same-time calls"
    local first_solver_time
    local second_solver_time
    first_solver_time=$(awk \
        '/Same-time closure call 0:/ {print $8}' \
        "$log")
    second_solver_time=$(awk \
        '/Same-time closure call 1:/ {print $8}' \
        "$log")
    [[ -n "$first_solver_time" \
        && "$first_solver_time" = "$second_solver_time" ]] \
        || fail "repeated closure calls did not preserve one solver time index"
    sed -n '1,100p' "$log"
}

run_uds_onnx_probe()
{
    echo "[FoamNordic] Verifying native ONNX inference over pure UDS"
    address="unix://$work_dir/closure-uds.sock"
    configure_probe U U U 1.0 0.25 false

    "$build_dir/tools/resident/foamnordic_model_worker" \
        "$address" \
        "$model_dir/identity-U.fnom" \
        --no-shm \
        >"$work_dir/worker-uds.log" 2>&1 &
    worker_pid=$!

    for _ in {1..50}; do
        [[ -S "$work_dir/closure-uds.sock" ]] && break
        sleep 0.1
    done
    [[ -S "$work_dir/closure-uds.sock" ]] \
        || fail "pure UDS closure worker did not create its endpoint"

    run_probe
    wait_bounded "$worker_pid" "pure UDS ONNX closure worker"
    worker_pid=
    [[ ! -e "$work_dir/closure-uds.sock" ]] \
        || fail "pure UDS closure worker left a stale endpoint"
    grep -Eq "Closure worker rank [0-9]+ data plane: UDS" \
        "$work_dir/worker-uds.log" \
        || fail "closure worker did not retain the UDS data plane"

    sed -n '1,100p' "$work_dir/worker-uds.log"
    echo "[FoamNordic] Native OpenFOAM ONNX pure UDS exchange: PASS"
}

run_mpi_probe()
{
    ((mpi_ranks > 1)) || return 0

    command -v decomposePar >/dev/null \
        || fail "required command is unavailable: decomposePar"
    command -v mpirun >/dev/null \
        || fail "required command is unavailable: mpirun"

    echo "[FoamNordic] Verifying $mpi_ranks rank-local OpenFOAM closures"
    address="unix://$work_dir/closure-{rank}.sock"
    configure_probe U U U 1.0 0.25 false

    sed \
        -e "s/@NUMBER_OF_SUBDOMAINS@/$mpi_ranks/g" \
        -e 's/@DECOMPOSITION_METHOD@/scotch/g' \
        -e 's/@METHOD_COEFFICIENTS@//g' \
        "$repository/tools/template/openfoam/decomposeParDict.in" \
        >"$case_dir/system/decomposeParDict"
    rm -rf "$case_dir"/processor*
    decomposePar -case "$case_dir" -force \
        >"$work_dir/decompose.log" 2>&1

    worker_pids=()
    local rank
    for ((rank = 0; rank < mpi_ranks; ++rank)); do
        "$build_dir/tools/openfoam/foamnordic_openfoam_echo" \
            "unix://$work_dir/closure-$rank.sock" \
            U \
            1.0 \
            >"$work_dir/rank-$rank.log" 2>&1 &
        worker_pids+=("$!")
    done
    wait_for_rank_endpoints

    local mpi_options=(-np "$mpi_ranks")
    if [[ -n "${SLURM_NTASKS:-}" \
        && "${SLURM_NTASKS}" -lt "$mpi_ranks" ]]; then
        if [[ "${SLURM_CPUS_PER_TASK:-1}" -lt "$mpi_ranks" ]]; then
            fail "the Slurm allocation has fewer CPUs than MPI ranks"
        fi
        mpi_options+=(--map-by :OVERSUBSCRIBE)
    fi

    mpirun \
        "${mpi_options[@]}" \
        "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
        -case "$case_dir" \
        -parallel \
        >"$work_dir/mpi-probe.log" 2>&1

    for pid in "${worker_pids[@]}"; do
        wait_bounded "$pid" "rank-local native echo worker"
    done
    worker_pids=()

    for ((rank = 0; rank < mpi_ranks; ++rank)); do
        local log="$work_dir/rank-$rank.log"
        grep -q "Data plane: SHM" "$log" \
            || fail "rank $rank did not negotiate SHM"
        [[ $(grep -c "Same-time closure call" "$log") -eq 2 ]] \
            || fail "rank $rank did not complete two closure calls"
        [[ $(awk '/Same-time closure call [01]:/ {print $8}' "$log" \
                | sort -u | wc -l) -eq 1 ]] \
            || fail "rank $rank did not preserve one solver time index"
    done

    grep -q "OpenFOAM closure hook: PASS" "$work_dir/mpi-probe.log" \
        || fail "parallel OpenFOAM closure probe did not report success"
    sed -n '1,120p' "$work_dir/mpi-probe.log"
    for ((rank = 0; rank < mpi_ranks; ++rank)); do
        sed -n '1,40p' "$work_dir/rank-$rank.log"
    done
    echo "[FoamNordic] Rank-local OpenFOAM closure exchange: PASS"
}

run_longship_mpi_onnx_probe()
{
    ((mpi_ranks > 1)) || return 0

    echo "[FoamNordic] Verifying one ONNX ModelHost with $mpi_ranks OpenFOAM ranks"
    address="unix://$work_dir/closure-shared.sock"
    configure_probe U U U 1.0 0.25 false

    local mpi_options=(-np "$mpi_ranks")
    if [[ -n "${SLURM_NTASKS:-}" \
        && "${SLURM_NTASKS}" -lt "$mpi_ranks" ]]; then
        if [[ "${SLURM_CPUS_PER_TASK:-1}" -lt "$mpi_ranks" ]]; then
            fail "the Slurm allocation has fewer CPUs than MPI ranks"
        fi
        mpi_options+=(--map-by :OVERSUBSCRIBE)
    fi

    "$build_dir/tools/longship/foamnordic-longship" \
        --ready "$work_dir/closure-shared.sock" \
        --host-output "$work_dir/longship-host.log" \
        --solver-output "$work_dir/longship-solver.log" \
        --readiness-timeout-ms "$((test_timeout * 1000))" \
        --termination-grace-ms 5000 \
        --host \
        "$build_dir/tools/resident/foamnordic_model_worker" \
        "$address" \
        "$model_dir/identity-U.fnom" \
        --connections "$mpi_ranks" \
        --solver \
        mpirun \
        "${mpi_options[@]}" \
        "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
        -case "$case_dir" \
        -parallel \
        >"$work_dir/longship.log" 2>&1

    [[ ! -e "$work_dir/closure-shared.sock" ]] \
        || fail "Longship left its shared ModelHost endpoint"
    [[ $(grep -Ec "Closure worker rank [0-9]+ data plane: SHM" \
            "$work_dir/longship-host.log") -eq "$mpi_ranks" ]] \
        || fail "shared ModelHost did not negotiate SHM with every rank"
    [[ $(grep -c "Native closure runner stopped" \
            "$work_dir/longship-host.log") -eq "$mpi_ranks" ]] \
        || fail "shared ModelHost did not reap every rank session"
    grep -q "OpenFOAM closure hook: PASS" "$work_dir/longship-solver.log" \
        || fail "Longship OpenFOAM probe did not report success"
    grep -q "Longship completed successfully" "$work_dir/longship.log" \
        || fail "Longship did not report coupled success"

    sed -n '1,100p' "$work_dir/longship.log"
    sed -n '1,120p' "$work_dir/longship-host.log"
    sed -n '1,120p' "$work_dir/longship-solver.log"
    echo "[FoamNordic] Shared ONNX ModelHost Longship exchange: PASS"
}

run_split_ucx_onnx_probe()
{
    [[ "${FOAMNORDIC_UCX_SPLIT:-false}" == true ]] || return
    [[ -n "${SLURM_JOB_ID:-}" ]] \
        || fail "FOAMNORDIC_UCX_SPLIT requires an interactive server allocation"
    for command in awk flock grep ip sacct sbatch scancel squeue; do
        command -v "$command" >/dev/null \
            || fail "required split-UCX command is unavailable: $command"
    done

    exec {split_lock_fd}>"$work_dir/.ucx-split.lock"
    flock -n "$split_lock_fd" \
        || fail "another central UCX gate is already using $work_dir"

    local account=${FOAMNORDIC_SLURM_ACCOUNT:?Set FOAMNORDIC_SLURM_ACCOUNT}
    local partition=${FOAMNORDIC_UCX_CLIENT_PARTITION:-small}
    local client_wait=${FOAMNORDIC_UCX_CLIENT_WAIT:-900}
    local central_ranks=${FOAMNORDIC_CENTRAL_MPI_RANKS:-1}
    [[ "$central_ranks" =~ ^[1-9][0-9]*$ ]] \
        || fail "FOAMNORDIC_CENTRAL_MPI_RANKS must be a positive integer"
    local server_node
    server_node=$(hostname -s)
    local interface=${FOAMNORDIC_UCX_INTERFACE:-ib0}
    local ucx_host=${FOAMNORDIC_UCX_HOST:-}
    if [[ -z "$ucx_host" ]]; then
        ucx_host=$(ip -o -4 address show dev "$interface" \
            | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}')
    fi
    [[ -n "$ucx_host" ]] \
        || fail "UCX interface $interface has no IPv4 address"
    local port=${FOAMNORDIC_UCX_CONTROL_PORT:-$((24000 + (SLURM_JOB_ID + $$) % 20000))}
    ((port <= 65535)) || fail "UCX control-plane port exceeds 65535"
    address="tcp://$server_node:$port"
    closure_shared_memory=false
    closure_ucx=true
    configure_probe U U U 1.0 0.25 false

    local parallel_options=()
    local client_prefix=()
    if ((central_ranks > 1)); then
        command -v decomposePar >/dev/null \
            || fail "decomposePar is required for the central MPI gate"
        command -v mpirun >/dev/null \
            || fail "mpirun is required for the central MPI gate"
        sed \
            -e "s/@NUMBER_OF_SUBDOMAINS@/$central_ranks/g" \
            -e 's/@DECOMPOSITION_METHOD@/scotch/g' \
            -e 's/@METHOD_COEFFICIENTS@//g' \
            "$repository/tools/template/openfoam/decomposeParDict.in" \
            >"$case_dir/system/decomposeParDict"
        rm -rf "$case_dir"/processor*
        decomposePar -case "$case_dir" -force \
            >"$work_dir/ucx-decompose.log" 2>&1
        client_prefix=("$(command -v mpirun)" -np "$central_ranks")
        parallel_options=(-parallel)
    fi

    local ready="$work_dir/ucx-worker.ready"
    local host_log="$work_dir/ucx-host.log"
    local client_log="$work_dir/ucx-client-%j.log"
    local client_job_file="$work_dir/ucx-client.job"
    local client_proxy_log="$work_dir/ucx-client-proxy.log"
    local longship_log="$work_dir/ucx-longship.log"
    export UCX_TLS=${FOAMNORDIC_UCX_TLS:-rc,ud,sm,self}
    rm -f "$ready"
    rm -f "$client_job_file"

    echo "[FoamNordic] Central UCX host allocation: $SLURM_JOB_ID"
    echo "[FoamNordic] Central UCX host node: $server_node"
    echo "[FoamNordic] Central UCX OpenFOAM ranks: $central_ranks"
    echo "[FoamNordic] Central UCX control address: $address"
    echo "[FoamNordic] Central UCX interface address: $ucx_host"
    echo "[FoamNordic] Central UCX transports: $UCX_TLS"

    local longship_status
    set +e
    "$build_dir/tools/longship/foamnordic-longship" \
        --ready "$ready" \
        --host-output "$host_log" \
        --solver-output "$client_proxy_log" \
        --readiness-timeout-ms "$((test_timeout * 1000))" \
        --termination-grace-ms 30000 \
        --host \
        "$build_dir/tools/resident/foamnordic_model_worker" \
        "$address" \
        "$model_dir/identity-U.fnom" \
        --connections "$central_ranks" \
        --no-shm \
        --ucx-host "$ucx_host" \
        --ready-file "$ready" \
        --solver \
        "$repository/tools/longship/runSlurmClient.sh" \
        --account "$account" \
        --partition "$partition" \
        --time 00:10:00 \
        --nodes 1 \
        --ntasks "$central_ranks" \
        --cpus-per-task 1 \
        --exclude "$server_node" \
        --job-name fn-of-ucx \
        --output "$client_log" \
        --job-id-file "$client_job_file" \
        --wait-seconds "$client_wait" \
        -- \
        "${client_prefix[@]}" \
        "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
        -case \
        "$case_dir" \
        "${parallel_options[@]}" \
        >"$longship_log" 2>&1
    longship_status=$?
    set -e
    if [[ "$longship_status" -ne 0 ]]; then
        sed -n '1,160p' "$longship_log" >&2
        sed -n '1,200p' "$host_log" >&2
        sed -n '1,160p' "$client_proxy_log" >&2
        fail "central UCX Longship exited with status $longship_status"
    fi

    [[ -s "$client_job_file" ]] \
        || fail "central UCX Slurm proxy did not record its client job"
    local client_job
    client_job=$(<"$client_job_file")
    client_log=${client_log/\%j/$client_job}
    echo "[FoamNordic] Central UCX client job: $client_job"
    [[ ! -e "$ready" ]] \
        || fail "central UCX Longship left its readiness marker"
    [[ $(grep -Ec "Closure worker rank [0-9]+ data plane: UCX" \
            "$host_log") -eq "$central_ranks" ]] \
        || fail "central ModelHost did not negotiate UCX with every rank"
    [[ $(grep -c "Native closure runner stopped" "$host_log") \
            -eq "$central_ranks" ]] \
        || fail "central ModelHost did not reap every rank session"
    grep -q "OpenFOAM closure hook: PASS" "$client_log" \
        || fail "central UCX OpenFOAM client did not report success"
    grep -q "Slurm client completed: $client_job" "$client_proxy_log" \
        || fail "central UCX Slurm proxy did not report completion"
    grep -q "Longship completed successfully" "$longship_log" \
        || fail "central UCX Longship did not report coupled success"

    sed -n '1,120p' "$longship_log"
    sed -n '1,120p' "$client_proxy_log"
    sed -n '1,140p' "$host_log"
    sed -n '1,180p' "$client_log"
    echo "[FoamNordic] Central $central_ranks-rank OpenFOAM ONNX ModelHost over UCX: PASS"
}

export FOAMNORDIC_SOURCE="$repository"
export FOAMNORDIC_BUILD="$build_dir"
export FOAMNORDIC_OPENFOAM_LIB="$work_dir/lib"
export FOAM_USER_LIBBIN="$work_dir/lib"
export FOAM_USER_APPBIN="$work_dir/bin"
export LD_LIBRARY_PATH="$ort_root/lib:$work_dir/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$ort_root/lib:$work_dir/lib:${DYLD_LIBRARY_PATH:-}"

mkdir -p "$FOAM_USER_LIBBIN" "$FOAM_USER_APPBIN"

echo "[FoamNordic] OpenFOAM: $WM_PROJECT_VERSION"
echo "[FoamNordic] ONNX Runtime: $(<"$ort_root/VERSION_NUMBER")"
echo "[FoamNordic] Work directory: $work_dir"

cmake_options=(
    -S "$repository" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFOAMNORDIC_TESTS=ON \
    -DFOAMNORDIC_OPENFOAM_TOOLS=ON \
    -DFOAMNORDIC_ONNX_RUNTIME=ON \
    -DFOAMNORDIC_ONNX_RUNTIME_ROOT="$ort_root" \
    -DFOAMNORDIC_RESIDENT_TOOLS=ON
)
if [[ -n "${FOAMNORDIC_UCX_ROOT:-}" ]]; then
    [[ -f "$FOAMNORDIC_UCX_ROOT/include/ucp/api/ucp.h" ]] \
        || fail "FOAMNORDIC_UCX_ROOT is not a UCX installation"
    cmake_options+=(
        -DFOAMNORDIC_UCX=ON
        -DFOAMNORDIC_UCX_ROOT="$FOAMNORDIC_UCX_ROOT"
    )
    export FOAMNORDIC_UCX_FLAGS="-DFOAMNORDIC_HAVE_UCX=1 -I$FOAMNORDIC_UCX_ROOT/include"
    export FOAMNORDIC_UCX_LIBS="-L$FOAMNORDIC_UCX_ROOT/lib -Wl,-rpath,$FOAMNORDIC_UCX_ROOT/lib -lucp -lucs"
elif [[ "${FOAMNORDIC_UCX_SPLIT:-false}" == true ]]; then
    fail "FOAMNORDIC_UCX_SPLIT requires FOAMNORDIC_UCX_ROOT"
fi

if [[ -n "$resume_work_dir" ]]; then
    [[ "${FOAMNORDIC_UCX_SPLIT:-false}" == true ]] \
        || fail "resume mode requires FOAMNORDIC_UCX_SPLIT=true"
    [[ -x "$build_dir/tools/resident/foamnordic_model_worker" ]] \
        || fail "resume work directory lacks the ModelHost executable"
    [[ -x "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" ]] \
        || fail "resume work directory lacks the OpenFOAM probe"
    [[ -f "$model_dir/identity-U.fnom" && -d "$case_dir/system" ]] \
        || fail "resume work directory lacks its model or OpenFOAM case"
    echo "[FoamNordic] Resuming the central UCX gate from existing artifacts"
    run_split_ucx_onnx_probe
    echo "[FoamNordic] Native OpenFOAM ONNX closure hook: PASS"
    exit 0
fi
cmake "${cmake_options[@]}"

cmake --build "$build_dir" --parallel "$jobs"
ctest --test-dir "$build_dir" --output-on-failure

cp -R "$repository/src/foamnordic/openfoam" "$work_dir/openfoam-adapter"
cp -R \
    "$repository/tools/openfoam/closureHookProbe" \
    "$work_dir/closure-hook-probe"

(
    cd "$work_dir/openfoam-adapter"
    wclean libso
    wmake libso
)
(
    cd "$work_dir/closure-hook-probe"
    wclean
    wmake
)

cp -R "$source_case" "$case_dir"
foamDictionary \
    "$case_dir/system/controlDict" \
    -entry startFrom \
    -set latestTime

echo "[FoamNordic] Verifying derived OpenFOAM closure input"
configure_probe p "laplacian(p)" p 1.0 0.0 false
run_echo_probe p 1.0 "$work_dir/derived.log"

echo "[FoamNordic] Verifying non-identity native field replacement"
configure_probe U U U 1.005 0.25 false
run_echo_probe U 1.005 "$work_dir/scaled.log"

if [[ -n "${FOAMNORDIC_TEST_PATCH:-}" ]]; then
    echo "[FoamNordic] Verifying boundary patch exchange"
    configure_probe \
        U U U 1.0 0.0 false \
        "$FOAMNORDIC_TEST_PATCH"
    run_echo_probe U 1.0 "$work_dir/patch.log"
fi

echo "[FoamNordic] Verifying atomic worker rejection"
configure_probe U U U 1.0 0.25 true
run_rejection_probe "$work_dir/rejected.log"

echo "[FoamNordic] Verifying native ONNX inference"
configure_probe U U U 1.0 0.25 false
"$build_dir/tools/openfoam/foamnordic_openfoam_onnx_fixture" "$model_dir"
"$build_dir/tools/resident/foamnordic_model_worker" \
    "$address" \
    "$model_dir/identity-U.fnom" \
    >"$work_dir/worker.log" 2>&1 &
worker_pid=$!
wait_for_endpoint
run_probe
wait_bounded "$worker_pid" "native ONNX closure worker"
worker_pid=
[[ ! -e "$work_dir/closure.sock" ]] \
    || fail "native closure worker left a stale UDS endpoint"

grep -Eq "Closure worker rank [0-9]+ data plane: SHM" "$work_dir/worker.log" \
    || fail "closure worker did not negotiate SHM"

sed -n '1,100p' "$work_dir/worker.log"

run_uds_onnx_probe
run_mpi_probe
run_longship_mpi_onnx_probe
run_split_ucx_onnx_probe

adapter_library=$(
    find "$FOAM_USER_LIBBIN" \
        -maxdepth 1 \
        -type f \
        -name 'libfoamnordicOpenFOAM.*' \
        -print \
        -quit
)
[[ -n "$adapter_library" ]] \
    || fail "OpenFOAM adapter library was not produced"

echo "[FoamNordic] SHA-256"
shasum -a 256 \
    "$adapter_library" \
    "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
    "$build_dir/tools/resident/foamnordic_model_worker" \
    "$model_dir/identity-U.fnom"

echo "[FoamNordic] Native OpenFOAM ONNX closure hook: PASS"
