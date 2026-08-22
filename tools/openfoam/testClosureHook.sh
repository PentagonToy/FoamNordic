#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNord] Closure hook test failed: $*" >&2
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
work_dir=$(mktemp -d "$work_parent/foamnordic-onnx-hook.XXXXXX")
build_dir="$work_dir/build"
case_dir="$work_dir/case"
model_dir="$work_dir/model"
address="unix://$work_dir/closure.sock"
worker_pid=
probe_pid=
watchdog_pid=
worker_pids=()

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

    cmake \
        -DINPUT="$repository/src/foamnordic/template/openfoam/closureDict.in" \
        -DOUTPUT="$case_dir/system/foamnordicClosureDict" \
        -DFOAMNORDIC_ADDRESS="$address" \
        -DFOAMNORDIC_SESSION_ID=1 \
        -DFOAMNORDIC_SHARED_MEMORY=true \
        -DFOAMNORDIC_INPUT_KEYS="$key" \
        -DFOAMNORDIC_INPUT_EXPRESSIONS="\"$expression\"" \
        -DFOAMNORDIC_OUTPUT_FIELDS="$output" \
        -DFOAMNORDIC_PROBE_EXPRESSION="$expression" \
        -DFOAMNORDIC_PROBE_OUTPUT="$output" \
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

run_mpi_probe()
{
    ((mpi_ranks > 1)) || return

    command -v decomposePar >/dev/null \
        || fail "required command is unavailable: decomposePar"
    command -v mpirun >/dev/null \
        || fail "required command is unavailable: mpirun"

    echo "[FoamNord] Verifying $mpi_ranks rank-local OpenFOAM closures"
    address="unix://$work_dir/closure-{rank}.sock"
    configure_probe U U U 1.0 0.25 false

    sed \
        -e "s/@NUMBER_OF_SUBDOMAINS@/$mpi_ranks/g" \
        -e 's/@DECOMPOSITION_METHOD@/scotch/g' \
        -e 's/@METHOD_COEFFICIENTS@//g' \
        "$repository/src/foamnordic/template/openfoam/decomposeParDict.in" \
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
    echo "[FoamNord] Rank-local OpenFOAM closure exchange: PASS"
}

export FOAMNORDIC_SOURCE="$repository"
export FOAMNORDIC_BUILD="$build_dir"
export FOAMNORDIC_OPENFOAM_LIB="$work_dir/lib"
export FOAM_USER_LIBBIN="$work_dir/lib"
export FOAM_USER_APPBIN="$work_dir/bin"
export LD_LIBRARY_PATH="$ort_root/lib:$work_dir/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$ort_root/lib:$work_dir/lib:${DYLD_LIBRARY_PATH:-}"

mkdir -p "$FOAM_USER_LIBBIN" "$FOAM_USER_APPBIN"

echo "[FoamNord] OpenFOAM: $WM_PROJECT_VERSION"
echo "[FoamNord] ONNX Runtime: $(<"$ort_root/VERSION_NUMBER")"
echo "[FoamNord] Work directory: $work_dir"

cmake \
    -S "$repository" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFOAMNORDIC_TESTS=ON \
    -DFOAMNORDIC_OPENFOAM_TOOLS=ON \
    -DFOAMNORDIC_ONNX_RUNTIME=ON \
    -DFOAMNORDIC_ONNX_RUNTIME_ROOT="$ort_root" \
    -DFOAMNORDIC_RESIDENT_TOOLS=ON

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

echo "[FoamNord] Verifying derived OpenFOAM closure input"
configure_probe p "laplacian(p)" p 1.0 0.0 false
run_echo_probe p 1.0 "$work_dir/derived.log"

echo "[FoamNord] Verifying non-identity native field replacement"
configure_probe U U U 1.005 0.25 false
run_echo_probe U 1.005 "$work_dir/scaled.log"

echo "[FoamNord] Verifying atomic worker rejection"
configure_probe U U U 1.0 0.25 true
run_rejection_probe "$work_dir/rejected.log"

echo "[FoamNord] Verifying native ONNX inference"
configure_probe U U U 1.0 0.25 false
"$build_dir/tools/openfoam/foamnordic_openfoam_onnx_fixture" "$model_dir"
"$build_dir/tools/resident/foamnordic_closure_worker" \
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

grep -q "Closure worker data plane: SHM" "$work_dir/worker.log" \
    || fail "closure worker did not negotiate SHM"

sed -n '1,100p' "$work_dir/worker.log"

run_mpi_probe

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

echo "[FoamNord] SHA-256"
shasum -a 256 \
    "$adapter_library" \
    "$FOAM_USER_APPBIN/foamnordicOpenFOAMClosureHookProbe" \
    "$build_dir/tools/resident/foamnordic_closure_worker" \
    "$model_dir/identity-U.onnx" \
    "$model_dir/identity-U.fnom"

echo "[FoamNord] Native OpenFOAM ONNX closure hook: PASS"
