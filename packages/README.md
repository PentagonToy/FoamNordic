# Package boundaries

**BindNordic understands the model. FoamNordic understands how that model is used in a simulation.**

This rule defines the architectural boundary of the workspace.

**FoamNordic** owns OpenFOAM semantics, field binding, MPI and Slurm placement, transport, packing, process lifecycle, observations, and atomic solver updates.

**BindNordic** owns FNOM, tensor contracts, scalers, model persistence, backend evaluation, compiled estimators, and model caches.

ModelHost and Smedja remain in FoamNordic because they depend on solver state and runtime lifecycle. Their model evaluation and scaler operations delegate to BindNordic. BindNordic tensors contain only model data and its name, dtype and shape; rank, exchange, session and simulation time remain FoamNordic context.

The dependency is one-way: FoamNordic may depend on BindNordic, but BindNordic must not depend on FoamNordic. Users continue to work through the FoamNordic API and do not need to import BindNordic directly.

## Placement rule

Code that remains meaningful without OpenFOAM, placement or simulation time belongs in BindNordic. Code that requires solver fields, ranks, exchange ordering or lifecycle belongs in FoamNordic. A change that crosses this boundary must document why the lower-level package cannot own the behaviour independently.
