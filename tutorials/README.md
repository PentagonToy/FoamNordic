# Tutorials

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

## Read an example online

Three guides show the main execution paths with copyable original notebook code:

| Guide | Execution | Purpose |
| --- | --- | --- |
| [Basic run](https://pentagontoy.github.io/FoamNordic/foamnordic/examples/lid-driven-cavity/) | Local | Launch an unchanged OpenFOAM case without a closure, transform or observation |
| [Field exchange](https://pentagontoy.github.io/FoamNordic/foamnordic/examples/field-exchange/) | Slurm | Transform velocity and observe fields during execution |
| [Smagorinsky closure](https://pentagontoy.github.io/FoamNordic/foamnordic/examples/smagorinsky-closure/) | Slurm | Couple a native symbolic closure through the LES adapter |

Each guide states environment requirements and expected outcomes. Cluster account, partition and resource settings must match the local scheduler. The catalogue below retains the other domain-specific and learned examples.

## Run a tutorial

Notebook examples locate the FoamNordic checkout from `Path.cwd()`. Start Jupyter anywhere inside the repository; each notebook reads its case from `tutorials/openfoam_tutorials/` and writes generated models and runs below `tutorials/foamnordic_tutorials/`.

Activate the Python environment and load OpenFOAM before running a notebook. Install `gdown` for the learned-closure tutorials that retrieve the prepared dataset from Google Drive. For an installation built from source, run `foamnordic build` once for the active OpenFOAM ABI. The notebooks use `blockMesh` through `case.initialize(...)`; generated meshes, time directories, processor directories, logs, models and run output are intentionally excluded from Git.

## Notebook catalogue

| Domain | Notebook | Purpose |
| --- | --- | --- |
| Combustion | [`openfoam_counter_flow_flame.ipynb`](foamnordic_tutorials/combustion/openfoam_counter_flow_flame.ipynb) | Run an unchanged laminar counter-flow flame case through Longship |
| Compressible | [`k_equation_pitz_daily.ipynb`](foamnordic_tutorials/compressible/k_equation_pitz_daily.ipynb) | Declare native `kEqnFjord` transport for `rhoPimpleFoam` |
| Incompressible | [`openfoam_lid_driven_cavity.ipynb`](foamnordic_tutorials/incompressible/openfoam_lid_driven_cavity.ipynb) | Establish an unchanged OpenFOAM baseline |
| Incompressible | [`field_exchange_lid_driven_cavity.ipynb`](foamnordic_tutorials/incompressible/field_exchange_lid_driven_cavity.ipynb) | Transform and observe fields through the generic exchange boundary |
| Incompressible | [`smagorinsky_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/smagorinsky_pitz_daily.ipynb) | Declare a native symbolic Smagorinsky closure |
| Incompressible | [`wale_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/wale_pitz_daily.ipynb) | Declare a native symbolic WALE closure |
| Incompressible | [`k_equation_naca4412.ipynb`](foamnordic_tutorials/incompressible/k_equation_naca4412.ipynb) | Declare native `kEqnFjord` transport for an external-aerodynamics case |
| Incompressible | [`extra_trees_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/extra_trees_pitz_daily.ipynb) | Train, export and couple an Extra Trees closure |
| Incompressible | [`equinox_dnn_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/equinox_dnn_pitz_daily.ipynb) | Train, export and couple an Equinox DNN closure |

The hierarchy remains organised by physical solver domain rather than by model framework. This keeps each notebook beside its OpenFOAM case family while the filename makes the mathematical or learned execution path explicit.
