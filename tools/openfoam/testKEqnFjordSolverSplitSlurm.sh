#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNord] kEqnFjord solver test failed: $*" >&2
    exit 1
}

for command in awk cmake decomposePar flock foamDictionary grep ip mpirun; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done
[[ -n "${WM_PROJECT_VERSION:-}" ]] \
    || fail "load an OpenFOAM environment before running this test"
[[ -n "${SLURM_JOB_ID:-}" ]] \
    || fail "run this test inside an interactive ClosureHost allocation"

repository=$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)
prepared=${FOAMNORDIC_PREPARED_WORK_DIR:?Set FOAMNORDIC_PREPARED_WORK_DIR}
source_case=${FOAMNORDIC_SOURCE_CASE:?Set FOAMNORDIC_SOURCE_CASE}
ort_root=${FOAMNORDIC_ONNX_RUNTIME_ROOT:?Set FOAMNORDIC_ONNX_RUNTIME_ROOT}
account=${FOAMNORDIC_SLURM_ACCOUNT:?Set FOAMNORDIC_SLURM_ACCOUNT}
partition=${FOAMNORDIC_UCX_CLIENT_PARTITION:-small}
ranks=${FOAMNORDIC_CENTRAL_MPI_RANKS:-2}
wait_seconds=${FOAMNORDIC_UCX_CLIENT_WAIT:-900}
work_parent=${FOAMNORDIC_TEST_ROOT:?Set FOAMNORDIC_TEST_ROOT}
end_time=${FOAMNORDIC_SOLVER_END_TIME:-0.003}
[[ "$ranks" =~ ^[1-9][0-9]*$ && "$wait_seconds" =~ ^[1-9][0-9]*$ ]] \
    || fail "rank and wait values must be positive integers"
[[ -d "$source_case/system" && -f "$source_case/0/U" ]] \
    || fail "FOAMNORDIC_SOURCE_CASE is not an OpenFOAM case starting at 0"
[[ -f "$ort_root/VERSION_NUMBER" ]] \
    || fail "FOAMNORDIC_ONNX_RUNTIME_ROOT is invalid"

build_dir="$prepared/build"
model_dir="$prepared/model"
worker="$build_dir/tools/resident/foamnordic_closure_worker"
fixture="$build_dir/tools/openfoam/foamnordic_openfoam_onnx_fixture"
longship="$build_dir/tools/longship/foamnordic-longship"
client_proxy="$repository/tools/longship/runSlurmClient.sh"
[[ -x "$worker" && -x "$fixture" && -x "$longship" && -x "$client_proxy" ]] \
    || fail "prepared native worker, fixture generator, Longship, or Slurm proxy is unavailable"

mkdir -p "$model_dir"
"$fixture" "$model_dir"
[[ -f "$model_dir/kEqnFjord.fnom" ]] \
    || fail "fixture generator did not produce kEqnFjord.fnom"

work_dir=$(mktemp -d "$work_parent/foamnordic-keqnfjord-solver.XXXXXX")
case_dir="$work_dir/case"
cp -R "$source_case" "$case_dir"

if [[ ! -f "$case_dir/0/k" ]]; then
    k_template=${FOAMNORDIC_K_FIELD_TEMPLATE:-$repository/src/foamnordic/template/openfoam/k.cavity.in}
    [[ -f "$k_template" ]] \
        || fail "the source case lacks 0/k and its fallback template is unavailable"
    cp "$k_template" "$case_dir/0/k"
    echo "[FoamNord] Seeded the compact cavity k field from: $k_template"
fi

set_entry()
{
    local file=$1
    local entry=$2
    local value=$3
    if foamDictionary "$file" -entry "$entry" >/dev/null 2>&1; then
        foamDictionary "$file" -entry "$entry" -set "$value" >/dev/null
    else
        foamDictionary "$file" -entry "$entry" -add "$value" >/dev/null
    fi
}

control="$case_dir/system/controlDict"
set_entry "$control" application pimpleFoam
set_entry "$control" libs '("libfoamnordicOpenFOAM.so")'
set_entry "$control" startFrom startTime
set_entry "$control" startTime 0
set_entry "$control" stopAt endTime
set_entry "$control" endTime "$end_time"
set_entry "$control" deltaT 0.001
set_entry "$control" adjustTimeStep false
set_entry "$control" writeControl timeStep
set_entry "$control" writeInterval 1

schemes="$case_dir/system/fvSchemes"
solution="$case_dir/system/fvSolution"
set_entry "$schemes" 'divSchemes/div(phi,k)' 'Gauss upwind'

echo "[FoamNord] k equation numerics"
foamDictionary "$schemes" -entry 'divSchemes/div(phi,k)'
# Resolve k read-only: the compact case's regex solver also covers kFinal.
# Writing a literal k through foamDictionary would match and mutate that regex.
foamDictionary "$solution" -entry 'solvers/k'

server_node=$(hostname -s)
interface=${FOAMNORDIC_UCX_INTERFACE:-ib0}
ucx_host=${FOAMNORDIC_UCX_HOST:-}
if [[ -z "$ucx_host" ]]; then
    ucx_host=$(ip -o -4 address show dev "$interface" \
        | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}')
fi
[[ -n "$ucx_host" ]] || fail "UCX interface $interface has no IPv4 address"
port=${FOAMNORDIC_UCX_CONTROL_PORT:-$((24000 + (SLURM_JOB_ID + $$) % 20000))}
((port <= 65535)) || fail "UCX control port exceeds 65535"
address="tcp://$server_node:$port"

cmake \
    -DINPUT="$repository/src/foamnordic/template/openfoam/turbulenceProperties.kEqnFjord.in" \
    -DOUTPUT="$case_dir/constant/turbulenceProperties" \
    -DFOAMNORDIC_ADDRESS="$address" \
    -DFOAMNORDIC_SESSION_ID=1 \
    -P "$repository/tools/openfoam/configureClosureModelCase.cmake"

sed \
    -e "s/@NUMBER_OF_SUBDOMAINS@/$ranks/g" \
    -e 's/@DECOMPOSITION_METHOD@/scotch/g' \
    -e 's/@METHOD_COEFFICIENTS@//g' \
    "$repository/src/foamnordic/template/openfoam/decomposeParDict.in" \
    >"$case_dir/system/decomposeParDict"
rm -rf "$case_dir"/processor*
decomposePar -case "$case_dir" -force >"$work_dir/decompose.log" 2>&1

export FOAMNORDIC_SOURCE="$repository"
export FOAMNORDIC_BUILD="$build_dir"
export FOAMNORDIC_OPENFOAM_LIB="$prepared/lib"
export FOAM_USER_LIBBIN="$prepared/lib"
export FOAM_USER_APPBIN="$prepared/bin"
export LD_LIBRARY_PATH="$ort_root/lib:$prepared/lib:${LD_LIBRARY_PATH:-}"
export UCX_TLS=${FOAMNORDIC_UCX_TLS:-rc,ud,sm,self}

exec {lock_fd}>"$prepared/.keqnfjord-solver.lock"
flock -n "$lock_fd" \
    || fail "another kEqnFjord solver gate is using $prepared"

ready="$work_dir/closure.ready"
host_log="$work_dir/host.log"
proxy_log="$work_dir/client-proxy.log"
client_log="$work_dir/client-%j.log"
job_file="$work_dir/client.job"
longship_log="$work_dir/longship.log"

echo "[FoamNord] kEqnFjord solver work directory: $work_dir"
echo "[FoamNord] ClosureHost allocation: $SLURM_JOB_ID ($server_node)"
echo "[FoamNord] OpenFOAM ranks: $ranks"
echo "[FoamNord] End time: $end_time"
echo "[FoamNord] Control address: $address"
echo "[FoamNord] UCX interface address: $ucx_host"

set +e
"$longship" \
    --ready "$ready" \
    --host-output "$host_log" \
    --solver-output "$proxy_log" \
    --readiness-timeout-ms 120000 \
    --termination-grace-ms 30000 \
    --host \
    "$worker" \
    "$address" \
    "$model_dir/kEqnFjord.fnom" \
    --connections "$ranks" \
    --no-shm \
    --ucx-host "$ucx_host" \
    --ready-file "$ready" \
    --solver \
    "$client_proxy" \
    --account "$account" \
    --partition "$partition" \
    --time 00:10:00 \
    --nodes 1 \
    --ntasks "$ranks" \
    --cpus-per-task 1 \
    --exclude "$server_node" \
    --job-name fn-keqnfjord \
    --output "$client_log" \
    --job-id-file "$job_file" \
    --wait-seconds "$wait_seconds" \
    -- \
    "$(command -v mpirun)" -np "$ranks" \
    "$(command -v pimpleFoam)" -case "$case_dir" -parallel \
    >"$longship_log" 2>&1
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
    sed -n '1,160p' "$longship_log" >&2
    sed -n '1,220p' "$host_log" >&2
    sed -n '1,180p' "$proxy_log" >&2
    [[ -s "$job_file" ]] \
        && sed -n '1,280p' "${client_log/\%j/$(<"$job_file")}" >&2
    fail "Longship solver run exited with status $status"
fi

[[ -s "$job_file" ]] || fail "Slurm proxy did not record the solver job"
client_job=$(<"$job_file")
client_log=${client_log/\%j/$client_job}
[[ $(grep -Ec "Closure worker rank [0-9]+ data plane: UCX" "$host_log") \
        -eq "$ranks" ]] \
    || fail "not every solver rank negotiated UCX"
[[ $(grep -c "Native closure runner stopped" "$host_log") -eq "$ranks" ]] \
    || fail "ClosureHost did not reap every solver rank"
grep -q "Selecting LES turbulence model kEqnFjord" "$client_log" \
    || fail "pimpleFoam did not select kEqnFjord"
grep -q "^Time = $end_time" "$client_log" \
    || fail "pimpleFoam did not reach the requested end time"
grep -q '^End$' "$client_log" \
    || fail "pimpleFoam did not finish cleanly"
grep -q "Longship completed successfully" "$longship_log" \
    || fail "Longship did not report coupled success"
[[ ! -e "$ready" ]] || fail "Longship left its readiness marker"

for field in k nut kProduction kDissipationCoeff; do
    [[ -f "$case_dir/processor0/$end_time/$field" ]] \
        || fail "pimpleFoam did not write $field at time $end_time"
done

sed -n '1,100p' "$longship_log"
sed -n '1,100p' "$proxy_log"
sed -n '1,180p' "$host_log"
sed -n '1,280p' "$client_log"
echo "[FoamNord] $ranks-rank pimpleFoam kEqnFjord ONNX over central UCX: PASS"
