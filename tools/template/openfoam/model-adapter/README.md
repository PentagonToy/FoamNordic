# Model-adapter scaffold

This directory is a copyable scaffold for a new equation-level OpenFOAM ML
closure. It is intentionally not part of the OpenFOAM build and cannot be
compiled as shipped. The files contain unresolved `@UPPER_SNAKE_CASE@` tokens
and `#error` guards so that a placeholder can never become an accidental
physical model.

Use the scaffold only after identifying the native equation entry point:

1. Name the solver-owned input and output fields and their dimensions.
2. Select the exact correction, equation assembly, or source-evaluation site.
3. Choose the appropriate OpenFOAM base type and runtime-selection namespace.
4. Copy `ModelAdapter.H.in` and `ModelAdapter.C.in` into
   `src/foamnordic/openfoam/models/<model>/` and replace every token.
5. Replace `@INVOKE_CLOSURE_AT_MODEL_SITE@` with one blocking
   `ClosureHook::invoke` call and only the field bindings required by the
   contract. Do not put inference, sockets, scaling, or lifecycle code there.
6. Copy `closureCoeffs.in` for the case-side field contract.
7. Add its field kinds to
   `python/foamnordic/contracts/openfoam_adapters.yaml`; do not hard-code
   adapter widths in Python orchestration code.
8. Register the concrete type in `makeClosureModels.C`, add its source to
   `Make/files`, and add a solver-integrated acceptance case.

Do not start from this scaffold for a time-step boundary `Transform`; that is
already served by `foamNordicExchange`. Do not use a turbulence model as a
surrogate hook for iLES momentum closure. An iLES stress or source belongs at
the momentum-equation assembly point, normally through a dedicated solver or
an appropriate `fvModel` integration.
