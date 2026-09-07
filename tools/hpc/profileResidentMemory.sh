#!/usr/bin/env bash
#SBATCH --job-name=fno-memory
#SBATCH --account=project_2015384
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=foamnordic-memory-%j.out
#SBATCH --error=foamnordic-memory-%j.err

set -Eeuo pipefail

script_repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
repo=${FOAMNORDIC_REPO:-${SLURM_SUBMIT_DIR:-$script_repo}}
build=${TMPDIR:-/tmp}/foamnordic-memory-${SLURM_JOB_ID:-local}

[[ -f "$repo/CMakeLists.txt" && -d "$repo/.git" ]] || {
    echo "FoamNordic repository not found: $repo" >&2
    echo "Submit from the repository root or set FOAMNORDIC_REPO" >&2
    exit 2
}

cleanup()
{
    find "$build" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$build" 2>/dev/null || true
}
trap cleanup EXIT

echo "Host: $(hostname)"
echo "Architecture: $(uname -m)"
echo "FoamNordic commit: $(git -C "$repo" rev-parse HEAD)"
echo "Allocation: ${SLURM_JOB_ID:-local}"

cmake -S "$repo" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFOAMNORDIC_TESTS=OFF \
    -DFOAMNORDIC_BENCHMARKS=ON
cmake --build "$build" \
    --target foamnordic_resident_memory_benchmark \
    --parallel "${SLURM_CPUS_PER_TASK:-2}"

executable=$build/tools/benchmarks/foamnordic_resident_memory_benchmark
for repetition in 1 2 3 4 5 6 7; do
    echo "BENCHMARK RUN $repetition"
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "$executable"
    else
        "$executable"
    fi
done
