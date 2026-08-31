# FoamNordic tutorials

The tutorial tree keeps executable notebooks separate from their source
OpenFOAM cases:

```text
tutorials/
├── foamnordic_tutorials/
│   ├── combustion/
│   ├── compressible/
│   └── incompressible/
└── openfoam_tutorials/
    ├── combustion/
    ├── compressible/
    └── incompressible/
```

Notebook examples assume that the repository is checked out at
`/scratch/<allocation-account>/<user>/Codes/FoamNordic`. Replace the two
placeholders with the local allocation and user path. The corresponding case
is then read from `tutorials/openfoam_tutorials/`, while generated models and
runs are written below `tutorials/foamnordic_tutorials/`.

Activate the Python environment and load OpenFOAM before running a notebook.
For an installation built from source, run `foamnordic build` once for the
active OpenFOAM ABI. The notebooks use `blockMesh` through
`case.initialize(...)`; generated meshes, time directories, processor
directories, logs, models, and run output are intentionally excluded from Git.
