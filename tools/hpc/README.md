# HPC validation

These tools exercise FoamNordic's solver-agnostic lifecycle, transport and OpenFOAM integration on Linux clusters. Domain-solver acceptance belongs to the repository that owns the corresponding equations.

Run interactive benchmark scripts only inside an existing allocation. Submit allocation-owning wrappers from a login node. Each script prints its output root so that reviewed evidence can be separated from disposable run data.

## Two-node attached Longship gate

The gate compares a two-node OpenFOAM baseline with the same decomposition using an identity field exchange. Use a shared scratch path for the output:

```console
export FOAMNORDIC_SLURM_ACCOUNT=<allocation-account>
export FOAMNORDIC_TEST_ROOT=/scratch/<allocation-account>/<user>/Tests/foamnordic-multinode
export FOAMNORDIC_SLURM_PARTITION=medium

module load openfoam/2512
python "$FOAMNORDIC_REPO/tools/hpc/testMultiNodeLongship.py"
```

The script requests two nodes and two OpenFOAM tasks, places one ModelHost on each node, and prints `Two-node Longship gate: PASS` only after final `U` and `p` parity succeeds. This gate passed on Roihu's full-node `medium` partition with exact field parity. Use `test` when two nodes are available; `small` accepts only one node. macOS exercises MPI and node-wrapper unit paths but cannot reproduce separate node-local shared-memory namespaces.

The direct OpenFOAM baseline helpers remain useful for site diagnostics and do not load FoamNordic's model runtime.

## WALE performance benchmark

Run the paired WALE benchmark inside a Roihu CPU interactive allocation:

```console
bash "$FOAMNORDIC_REPO/tools/hpc/benchmarkWaleInteractive.sh"
```

The benchmark uses one warm-up followed by seven measured 400-step runs for each path. Their order alternates to reduce systematic node and temperature bias. Raw runs stay under the test output root until the result has been reviewed; only the report and useful log should then be archived with the curated benchmark records.

## NACA4412 native transport benchmark

Run the combined native kEqnFjord performance and numerical trajectory benchmark inside a Roihu CPU interactive allocation:

```console
bash "$FOAMNORDIC_REPO/tools/hpc/benchmarkNacaNativeInteractive.sh"
```

The first stage performs one warm-up and seven alternating 400-step measurements. The second stage restarts both paths independently from the same initial condition at 1, 10, 100 and 400 steps, then records field errors for `U`, `p`, `k` and `nut`.

## Complete OpenFOAM benchmark suite

Submit all documented `nutFjord` and `kEqnFjord` measurements from a Roihu login node:

```console
bash "$FOAMNORDIC_REPO/tools/hpc/submitOpenfoamBenchmarks.sh"
```

The submitted job uses the `interactive` partition and writes one Slurm log under the benchmark output directory. It measures Smagorinsky and WALE through `nutFjord`, validates both at 1, 10, 100 and 400 steps, then measures and validates native `kEqnFjord` on NACA4412.
