# Execution lifecycle
A Longship compiles declarations, prepares an isolated case, launches native processes, and returns a non-blocking Run.

```text
declare -> validate -> copy -> prepare -> launch -> observe -> stop
                                   |                         |
                                   +------ fail together ----+
```

## Local and Slurm

Without a scheduler, one rank launches directly and multiple ranks use the MPI launcher associated with the case environment. Slurm runs declare solver and model resources separately and place a worker on every solver node.

## Ownership and shutdown

Python owns a workload by default. run.stop waits normally; force=True requests bounded termination. Closing the owner cancels its workload unless run.detach explicitly transfers responsibility. Solver and worker failures tear down the peer, while logs remain in the run directory.

## Runtime generations

Builds publish immutable content-addressed generations and atomically select the current one. A run pins its exact generation. Cleanup preserves current, recent, pinned, incomplete, and damaged generations and removes only old complete unpinned assets.

## Durable result

A terminal Result contains status, timing, job identity, logs, isolated case, observations, and plan digest. Postprocessing opens the run-owned case.
