# FoamNordic templates

Templates are grouped by the component that consumes them:

- `openfoam/` contains dictionaries and dictionary fragments copied into an
  isolated case. `derivedSchemes.json` contains conservative fallback schemes
  for ML input expressions; they are added only when neither an exact scheme
  nor a usable section default exists in the copied case. Its
  `model-adapter/` directory is a deliberately non-compilable scaffold for a
  new equation-level OpenFOAM closure. `combustion-model/` adds a guarded
  progress-variable, reaction-rate, and beta-FDF manifold scaffold whose
  placeholders force each solver family to declare field ownership and native
  correction order.
- `slurm/` contains scheduler scripts. Solver output belongs in
  `logs/Sailing_<name>_<jobid>.out`; FoamNordic lifecycle output belongs in
  `logs/Sailing_<name>_<jobid>.log`.
- `shell/` contains local execution and environment wrappers.
- `large_banner.txt` and `small_banner.txt` are presentation assets shared by
  generated logs and terminal commands.

Use `@UPPER_SNAKE_CASE@` for substitutions. Every token must be resolved before
the generated file is executed. Keep site-specific modules, accounts,
partitions, absolute paths, model names, and field contracts out of templates;
they belong in the Python declaration or runtime profile. Add a new template
when a solver family needs a structurally different dictionary instead of
growing conditional string assembly in the launcher.

Repository examples and generated templates use placeholders such as
`<allocation-account>`, `<user>`, `<jobid>`, and `<compute-node>`. Do not commit
personal names, email addresses, project numbers, home or scratch paths, node
names, or fabric addresses. Package-author metadata is the deliberate
exception and belongs only in `python/pyproject.toml`.

Templates may describe pure OpenFOAM or model-coupled runs. A template must not
silently enable a closure, change physical boundary conditions, or mutate the
source case.

Observation paths must contain `{rank}` whenever a case may run with multiple
MPI ranks. They are generated below the marked run directory so `foamnordic
clobber --workspace ...` can remove the whole owned run without guessing which
files belong to FoamNordic.
