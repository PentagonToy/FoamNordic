# Joblib connector

Joblib models are loaded once by a managed Python resident. Prediction accepts
and returns the same packed matrices as the native ONNX connector; estimator
and scikit-learn details do not enter Fjord or the solver adapter.
