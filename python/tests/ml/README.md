# ML backend smoke tests

These tests deliberately use tiny deterministic models. They validate model
packaging and evaluation contracts rather than training quality:

- a scikit-learn `VotingRegressor` exported through Joblib with fitted input
  scaling;
- a small Equinox MLP reconstructed and evaluated by the resident backend;
- a real ONNX graph checked, exported, reloaded, and evaluated numerically.

Keep datasets and estimator sizes small enough for ordinary wheel CI. Physical
LES and combustion acceptance belongs to the OpenFOAM/HPC gates rather than
this directory.
