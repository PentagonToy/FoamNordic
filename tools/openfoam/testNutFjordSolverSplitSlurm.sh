#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNordic] nutFjord solver test failed: $*" >&2
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
[[ "$ranks" =~ ^[1-9][0-9]*$ && "$wait_seconds" =~ ^[1-9][0-9]*$ ]] \
    || fail "rank and wait values must be positive integers"
[[ -d "$source_case/system" && -f "$source_case/0/nut" ]] \
    || fail "FOAMNORDIC_SOURCE_CASE is not the prepared LES cavity"
[[ -f "$ort_root/VERSION_NUMBER" ]] \
    || fail "FOAMNORDIC_ONNX_RUNTIME_ROOT is invalid"

build_dir="$prepared/build"
model_dir="$prepared/model"
worker="$build_dir/tools/resident/foamnordic_closure_worker"
longship="$build_dir/tools/longship/foamnordic-longship"
client_proxy="$repository/tools/longship/runSlurmClient.sh"
[[ -x "$worker" && -x "$longship" && -x "$client_proxy" ]] \
    || fail "prepared native worker, Longship, or Slurm proxy is unavailable"
[[ -f "$model_dir/nutFjord.fnom" ]] \
    || fail "prepared work directory lacks nutFjord.fnom"

work_dir=$(mktemp -d "$work_parent/foamnordic-nutfjord-solver.XXXXXX")
case_dir="$work_dir/case"
cp -R "$source_case" "$case_dir"

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
set_entry "$control" endTime 0.003
set_entry "$control" deltaT 0.001
set_entry "$control" adjustTimeStep false

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
    -DINPUT="$repository/src/foamnordic/template/openfoam/turbulenceProperties.nutFjord.in" \
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

exec {lock_fd}>"$prepared/.nutfjord-solver.lock"
flock -n "$lock_fd" \
    || fail "another nutFjord solver gate is using $prepared"

ready="$work_dir/closure.ready"
host_log="$work_dir/host.log"
proxy_log="$work_dir/client-proxy.log"
client_log="$work_dir/client-%j.log"
job_file="$work_dir/client.job"
longship_log="$work_dir/longship.log"

echo "[FoamNordic] nutFjord solver work directory: $work_dir"
echo "[FoamNordic] ClosureHost allocation: $SLURM_JOB_ID ($server_node)"
echo "[FoamNordic] OpenFOAM ranks: $ranks"
echo "[FoamNordic] Control address: $address"
echo "[FoamNordic] UCX interface address: $ucx_host"

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
    "$model_dir/nutFjord.fnom" \
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
    --job-name fn-nutfjord \
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
        && sed -n '1,260p' "${client_log/\%j/$(<"$job_file")}" >&2
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
grep -q "Selecting LES turbulence model nutFjord" "$client_log" \
    || fail "pimpleFoam did not select nutFjord"
grep -q '^Time = 0.003' "$client_log" \
    || fail "pimpleFoam did not reach the requested end time"
grep -q '^End$' "$client_log" \
    || fail "pimpleFoam did not finish cleanly"
grep -q "Longship completed successfully" "$longship_log" \
    || fail "Longship did not report coupled success"
[[ ! -e "$ready" ]] || fail "Longship left its readiness marker"

sed -n '1,100p' "$longship_log"
sed -n '1,100p' "$proxy_log"
sed -n '1,180p' "$host_log"
sed -n '1,260p' "$client_log"
echo "[FoamNordic] $ranks-rank pimpleFoam nutFjord ONNX over central UCX: PASS"
