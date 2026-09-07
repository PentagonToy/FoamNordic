# OpenFOAM

No post-refactor OpenFOAM performance result has been accepted yet. Software unit and build gates are recorded in the benchmark [index](README.md); this page is reserved for actual solver and coupled-runtime evidence.

The first comparison will use one fixed case and allocation for both the stock and FoamNordic-coupled paths. It will record:

- commit, OpenFOAM ABI, node class, ranks, threads, mesh and time-step count;
- solver execution and wall time, Longship total time, and model inference time;
- peak resident memory for solver and ModelHost;
- exchange count, payload size, transport, artefact size and model backend;
- final-field numerical differences and declared tolerances.

At least one warm-up run precedes repeated measured runs. Alternating baseline and coupled order is preferred to reduce node-state and filesystem-cache bias. The release comparison remains coupled solver wall time divided by uncoupled solver wall time under identical conditions; orchestration is reported separately rather than folded into that ratio.
