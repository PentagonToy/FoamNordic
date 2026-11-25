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

Examples expose case, closure, observation and scheduler choices while omitting redundant defaults. `longship.launch(verbose=None)` inherits the workload verbosity; advanced launch options remain documented in the [execution reference](../docs/api/run-control.md). Figure styling uses Hedgehogs helpers on Matplotlib axes.

Each guide states environment requirements and expected outcomes. Cluster account, partition and resource settings must match the local scheduler. The catalogue below retains the other domain-specific and learned examples.

## Run a tutorial

Notebook examples locate the FoamNordic checkout from `Path.cwd()`. Start Jupyter anywhere inside the repository; repository-based notebooks read their cases from `tutorials/openfoam_tutorials/` and write generated models and runs below `tutorials/foamnordic_tutorials/`.

The `esi_` notebooks use the bundled OpenCFD cases. The `foundation_` notebooks use Foundation 14 and copy the source cases from `$FOAM_TUTORIALS` into `output/`. Load `openfoam-org/14` before starting their notebook kernel and use the `of14` branch.

Activate the Python environment and load OpenFOAM before running a notebook. Install `gdown` for the learned-closure tutorials that retrieve the prepared dataset from Google Drive. For an installation built from source, run `foamnordic build` once for the active OpenFOAM ABI. Examples that generate a mesh use `blockMesh` through `case.initialize(...)`; generated meshes, time directories, processor directories, logs, models and run output are intentionally excluded from Git.

## Notebook catalogue

| Domain | Notebook | Purpose |
| --- | --- | --- |
| Combustion | [`esi_openfoam_counter_flow_flame.ipynb`](foamnordic_tutorials/combustion/esi_openfoam_counter_flow_flame.ipynb) | Run an unchanged laminar counter-flow flame case through Longship |
| Compressible | [`esi_k_equation_pitz_daily.ipynb`](foamnordic_tutorials/compressible/esi_k_equation_pitz_daily.ipynb) | Declare native `kEqnFjord` transport for `rhoPimpleFoam` |
| Incompressible | [`foundation_airfoil2d.ipynb`](foamnordic_tutorials/incompressible/foundation_airfoil2d.ipynb) | Run the Foundation 14 airFoil2D baseline with its supplied mesh |
| Incompressible | [`foundation_channel395.ipynb`](foamnordic_tutorials/incompressible/foundation_channel395.ipynb) | Run the Foundation 14 channel with four MPI ranks using built-in WALE or native `nutFjord` |
| Incompressible | [`esi_openfoam_lid_driven_cavity.ipynb`](foamnordic_tutorials/incompressible/esi_openfoam_lid_driven_cavity.ipynb) | Establish an unchanged OpenFOAM baseline |
| Incompressible | [`esi_field_exchange_lid_driven_cavity.ipynb`](foamnordic_tutorials/incompressible/esi_field_exchange_lid_driven_cavity.ipynb) | Transform and observe fields through the generic exchange boundary |
| Incompressible | [`esi_smagorinsky_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/esi_smagorinsky_pitz_daily.ipynb) | Declare a native symbolic Smagorinsky closure |
| Incompressible | [`esi_wale_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/esi_wale_pitz_daily.ipynb) | Declare a native symbolic WALE closure |
| Incompressible | [`esi_k_equation_naca4412.ipynb`](foamnordic_tutorials/incompressible/esi_k_equation_naca4412.ipynb) | Declare native `kEqnFjord` transport for an external-aerodynamics case |
| Incompressible | [`esi_extra_trees_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/esi_extra_trees_pitz_daily.ipynb) | Train, export and couple an Extra Trees closure |
| Incompressible | [`esi_equinox_dnn_pitz_daily.ipynb`](foamnordic_tutorials/incompressible/esi_equinox_dnn_pitz_daily.ipynb) | Train, export and couple an Equinox DNN closure |

### Foundation 14 sample results

The Foundation notebooks include saved output from short runs on Roihu's `interactive` partition. Replace the account placeholder and select a partition before running them. The channel preview shows built-in WALE. Set `USE_FOAMNORDIC_WALE = True` to use the native `nutFjord` closure.

| Case | WALE implementation | MPI ranks | Final output | Roihu job |
| --- | --- | ---: | --- | ---: |
| airFoil2D | — | 1 | Iteration 5 | 3416545 |
| channel395 | Foundation | 4 | $t=1.0$ s | 3416581 |
| channel395 | FoamNordic `nutFjord` | 4 | $t=1.0$ s | 3416603 |
