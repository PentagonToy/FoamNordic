# HPC validation tools

These tools exercise FoamNordic's solver-agnostic lifecycle and transport on
Linux clusters. Domain-solver acceptance belongs to the repository that owns
the corresponding equations.

## Two-node attached Longship gate

The gate compares a two-node OpenFOAM baseline with the same decomposition
using an identity field exchange. Use a shared scratch path for the output:

```bash
export FOAMNORDIC_SLURM_ACCOUNT=<allocation-account>
export FOAMNORDIC_TEST_ROOT=/scratch/<allocation-account>/<user>/Tests/foamnordic-multinode
export FOAMNORDIC_SLURM_PARTITION=medium

module load openfoam/2512
python "$FOAMNORDIC_REPO/tools/hpc/testMultiNodeLongship.py"
```

The script requests two nodes and two OpenFOAM tasks, places one ModelHost on
each node, and prints `Two-node Longship gate: PASS` only after final `U` and
`p` parity succeeds. This gate passed on Roihu's full-node `medium` partition
with exact field parity. Use `test` when two nodes are available; `small`
accepts only one node. macOS exercises MPI and node-wrapper unit paths but
cannot reproduce separate node-local shared-memory namespaces.

The pure OpenFOAM baseline helpers remain useful for site diagnostics and do
not load FoamNordic's model runtime.
