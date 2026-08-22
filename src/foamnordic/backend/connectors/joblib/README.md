# Joblib connector

Joblib models are loaded once by a managed Python resident. Prediction accepts
and returns the same packed matrices as the native ONNX connector; estimator
and scikit-learn details do not enter Fjord or the solver adapter.

Export uses an uncompressed sibling payload so large NumPy-backed estimators
can be opened with `mmap_mode="r"`. FoamNordic never embeds the payload in FNOM,
reads it into an intermediate `bytes` value, or writes it through a second
temporary archive. Joblib is pickle-based and must only load trusted models.
