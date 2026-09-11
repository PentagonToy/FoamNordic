# BindNordic

BindNordic is FoamNordic's internal model and file-format package. It owns FNOM containers, tensor contracts, scaler execution, model loading and evaluation, compiled kernels, and focused OpenFOAM field decoding.

The authoritative [package boundary](../README.md) defines its scope: BindNordic understands the model, while FoamNordic understands how that model is used in a simulation. BindNordic must remain independently testable without importing FoamNordic, but it is developed, documented, versioned, and distributed as part of the FoamNordic repository.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/bindnordic/models/` | FNOM, payloads, model contracts, runtimes, and compilation |
| `src/bindnordic/native/` | Native scaler and model kernels |
| `src/bindnordic/openfoam/` | Stored OpenFOAM field decoding only |
| `src/bindnordic/scalers.py` | Fitted-scaler normalization |
| `tests/` | Package-boundary and model-runtime verification |

## Development

Install FoamNordic from the repository root; the resulting FoamNordic distribution includes BindNordic. Run the Python package tests with:

```console
python -m pytest
```

The repository-level pytest configuration runs BindNordic and FoamNordic together while keeping their same-named test modules isolated. Run BindNordic's native tests through the root FoamNordic CMake build. Public installation and usage instructions belong to the root [FoamNordic README](../../README.md).

Do not add release notes, publishing helpers, product assets, or duplicated architecture documentation here. Those responsibilities belong to the FoamNordic repository root.
