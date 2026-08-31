# Combustion HPC acceptance bundle

These scripts validate the combustion paths that have local OpenFOAM v2606
evidence but still require Linux HPC/OpenFOAM v2512 evidence. Existing Roihu
TCP, UCX, ONNX, `nutFjord`, and `kEqnFjord` records are not repeated.

Required variables:

```bash
export FOAMNORDIC_REPO=/path/to/FoamNordic
export FOAMNORDIC_VENV=/path/to/virtual/environment
export FOAMNORDIC_HPC_GATE_ROOT=/path/to/test/bundle
export FOAMNORDIC_SLURM_ACCOUNT=<allocation-account>
```

Prepare once from a login node:

```bash
bash "$FOAMNORDIC_REPO/tools/hpc/prepareCombustionGates.sh"
```

Submit the one-node, two-rank acceptance suite:

```bash
bash "$FOAMNORDIC_REPO/tools/hpc/submitCombustionGates.sh"
```

To run inside an existing interactive allocation, request at least two tasks
and execute:

```bash
bash "$FOAMNORDIC_REPO/tools/hpc/runCombustionGates.sh"
```

Submit the OpenFOAM-only baseline separately. It loads OpenFOAM, copies the
same reduced case, and runs stock `reactingFoam` without importing FoamNordic,
starting a resident worker, or loading the FoamNordic OpenFOAM library:

```bash
bash "$FOAMNORDIC_REPO/tools/hpc/submitPureOpenFOAMBaseline.sh"
```

The default `FOAMNORDIC_PURE_END_TIME=1` matches the stock adapter gates. Set
another end time explicitly for a longer trajectory.

The run writes one immutable result directory and creates `PASS` only after all
five gates succeed. A cross-node central-host campaign remains a separate
sixth gate. On sites with one-node development partitions, the preferred
topology is an existing `interactive` host allocation plus a submitted
one-node `small` solver allocation, with the host node excluded from the
client request. It does not require one two-node allocation.

After the four serial gates have passed, rerun only the MPI gate with:

```bash
FOAMNORDIC_GATE_SET=mpi FOAMNORDIC_SKIP_BUILD=true \
    bash "$FOAMNORDIC_REPO/tools/hpc/submitCombustionGates.sh"
```

`FOAMNORDIC_SKIP_BUILD=true` is intended only for a Python-only fix after the
same checkout and OpenFOAM ABI have already completed `foamnordic build`.
