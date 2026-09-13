# ML backend smoke tests

These tests use small deterministic models to validate packaging and evaluation contracts rather than training quality.

| Backend | Coverage |
| --- | --- |
| Joblib | A scikit-learn `VotingRegressor` with fitted input scaling |
| Equinox | A small MLP reconstructed and evaluated by the resident backend |
| ONNX | A real graph checked, exported, reloaded and evaluated numerically |

Keep datasets and estimator sizes small enough for ordinary wheel CI. Physical LES and combustion acceptance belongs to the OpenFOAM/HPC gates rather than this directory.
