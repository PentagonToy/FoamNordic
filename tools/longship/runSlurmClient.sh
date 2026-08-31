#!/usr/bin/env bash

set -Eeuo pipefail

fail()
{
    echo "[FoamNordic] Slurm client failed: $*" >&2
    exit 1
}

usage()
{
    cat >&2 <<'EOF'
Usage: runSlurmClient.sh [OPTIONS] -- COMMAND [ARG ...]
  --account NAME
  --partition NAME
  --time HH:MM:SS
  --nodes N
  --ntasks N
  --cpus-per-task N
  --exclude HOSTS
  --job-name NAME
  --output PATH
  --job-id-file PATH
  --wait-seconds N
EOF
}

account=
partition=
time_limit=00:10:00
nodes=1
ntasks=1
cpus_per_task=1
exclude=
job_name=foamnordic-client
output=slurm-%j.out
job_id_file=
wait_seconds=900

while (($# > 0)); do
    case "$1" in
        --account) account=${2:?}; shift 2 ;;
        --partition) partition=${2:?}; shift 2 ;;
        --time) time_limit=${2:?}; shift 2 ;;
        --nodes) nodes=${2:?}; shift 2 ;;
        --ntasks) ntasks=${2:?}; shift 2 ;;
        --cpus-per-task) cpus_per_task=${2:?}; shift 2 ;;
        --exclude) exclude=${2:?}; shift 2 ;;
        --job-name) job_name=${2:?}; shift 2 ;;
        --output) output=${2:?}; shift 2 ;;
        --job-id-file) job_id_file=${2:?}; shift 2 ;;
        --wait-seconds) wait_seconds=${2:?}; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[[ -n "$account" && -n "$partition" ]] \
    || fail "--account and --partition are required"
(($# > 0)) || fail "a client command is required after --"
for value in "$nodes" "$ntasks" "$cpus_per_task" "$wait_seconds"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] \
        || fail "node, task, CPU, and wait values must be positive integers"
done
for command in awk grep sacct sbatch scancel squeue; do
    command -v "$command" >/dev/null \
        || fail "required command is unavailable: $command"
done

client_command=exec
for argument in "$@"; do
    printf -v client_command '%s %q' "$client_command" "$argument"
done

client_job=
completed=false
cleanup()
{
    if [[ "$completed" != true && -n "$client_job" ]]; then
        scancel "$client_job" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' INT TERM

sbatch_options=(
    --parsable
    --account="$account"
    --partition="$partition"
    --time="$time_limit"
    --nodes="$nodes"
    --ntasks="$ntasks"
    --cpus-per-task="$cpus_per_task"
    --job-name="$job_name"
    --output="$output"
    --export=ALL
)
if [[ -n "$exclude" ]]; then
    sbatch_options+=(--exclude="$exclude")
fi

client_job=$(sbatch "${sbatch_options[@]}" --wrap="$client_command")
if [[ -n "$job_id_file" ]]; then
    printf '%s\n' "$client_job" >"$job_id_file"
fi
echo "[FoamNordic] Slurm client job: $client_job"

deadline=$((SECONDS + wait_seconds))
while squeue --noheader --jobs="$client_job" | grep -q .; do
    ((SECONDS < deadline)) \
        || { echo "[FoamNordic] Slurm client exceeded ${wait_seconds}s" >&2; exit 124; }
    sleep 1
done

state=
for _ in {1..30}; do
    state=$(sacct --noheader --allocations --jobs="$client_job" \
        --format=State | awk 'NF {print $1; exit}')
    [[ -n "$state" ]] && break
    sleep 1
done
[[ "$state" == COMPLETED* ]] \
    || fail "job $client_job ended in state ${state:-unknown}"

completed=true
echo "[FoamNordic] Slurm client completed: $client_job"
