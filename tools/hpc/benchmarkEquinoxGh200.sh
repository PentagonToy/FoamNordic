#!/usr/bin/env bash
#SBATCH --job-name=fno-eqx-cold
#SBATCH --account=project_2015384
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2015384/Hanseul/Tests/foamnordic-equinox-%j.out
#SBATCH --error=/scratch/project_2015384/Hanseul/Tests/foamnordic-equinox-%j.err

set -Eeuo pipefail

repo=${FOAMNORDIC_REPO:-${SLURM_SUBMIT_DIR:-}}
venv=${FOAMNORDIC_GPU_VENV:-/scratch/project_2015384/Hanseul/Tests/venvs/foamnordic-gpu-py312}

[[ -f "$repo/tools/benchmarks/equinox_runtime.py" ]] || {
    echo "Submit from the FoamNordic repository root or set FOAMNORDIC_REPO" >&2
    exit 2
}
[[ -x "$venv/bin/python" ]] || {
    echo "GPU Python environment not found: $venv" >&2
    exit 2
}

export PATH="$venv/bin:$PATH"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset LD_LIBRARY_PATH

echo "Host: $(hostname)"
echo "FoamNordic commit: $(git -C "$repo" rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$venv/bin/python" - <<'PY'
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not devices or any(device.platform != "gpu" for device in devices):
    raise RuntimeError("The allocated GH200 is not the exclusive JAX backend")
PY

"$venv/bin/python" "$repo/tools/benchmarks/equinox_runtime.py"
