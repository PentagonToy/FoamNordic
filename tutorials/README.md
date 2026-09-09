# FoamNordic tutorials

The tutorial tree keeps executable notebooks separate from their source OpenFOAM cases:

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

Notebook examples locate the FoamNordic checkout from `Path.cwd()`. Start Jupyter anywhere inside the repository; each notebook reads its case from `tutorials/openfoam_tutorials/` and writes generated models and runs below `tutorials/foamnordic_tutorials/`.

Activate the Python environment and load OpenFOAM before running a notebook. For an installation built from source, run `foamnordic build` once for the active OpenFOAM ABI. The notebooks use `blockMesh` through `case.initialize(...)`; generated meshes, time directories, processor directories, logs, models, and run output are intentionally excluded from Git.
