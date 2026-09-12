from __future__ import annotations

import errno
import inspect
import sys
import tempfile
import types
from io import BytesIO
from pathlib import Path

import pytest
from foamnordic.core.native_plan import available as native_available

import foamnordic as fno


class _LinearPrediction:
    def __init__(self):
        import numpy as np

        self.weights = np.asarray([2.0, 1.0], dtype=np.float64)

    def predict(self, values):
        return (values @ self.weights).reshape(-1, 1)


def test_joblib_file_limit_falls_back_to_buffered_loading(mocker) -> None:
    from bindnordic.models.runtime import _load_joblib

    model = object()
    joblib = mocker.Mock()
    joblib.load.side_effect = [OSError(errno.EMFILE, "too many open files"), model]

    loaded, storage, mode = _load_joblib(Path("model.joblib"), joblib)

    assert loaded is model
    assert storage is None
    assert mode == "buffered"
    assert joblib.load.call_count == 2
    assert joblib.load.call_args_list[0].kwargs == {"mmap_mode": "r"}
    assert joblib.load.call_args_list[1].kwargs == {}


def test_joblib_non_file_limit_oserror_is_not_hidden(mocker) -> None:
    from bindnordic.models.runtime import _load_joblib

    joblib = mocker.Mock()
    joblib.load.side_effect = OSError(errno.EIO, "input/output error")

    with pytest.raises(OSError) as raised:
        _load_joblib(Path("model.joblib"), joblib)

    assert raised.value.errno == errno.EIO
    assert joblib.load.call_count == 1


def test_staged_joblib_releases_storage_after_buffered_fallback(mocker) -> None:
    from bindnordic.models.runtime import EmbeddedPayload, _staged_joblib

    model = object()
    embedded = EmbeddedPayload(Path("model.fnom"), 64, 128)
    extract = mocker.patch("bindnordic.models.runtime.extract_payload")
    _mock_patch_2 = mocker.patch(
        "bindnordic.models.runtime._load_joblib_file",
        return_value=(model, False),
    )
    try:
        loaded, storage, mode = _staged_joblib(embedded, mocker.Mock())
    finally:
        mocker.stop(_mock_patch_2)
        mocker.stop(extract)

    assert loaded is model
    assert storage is None
    assert mode == "buffered"
    extract.assert_called_once()


def test_export_namespace_exposes_exporters() -> None:
    assert dir(fno.Export) == ["Tensor", "equinox", "onnx", "sklearn", "table"]
    assert not hasattr(fno.Export, "compiled")
    assert not hasattr(fno.Export, "joblib")


def test_export_help_is_backend_discoverable() -> None:
    assert dir(fno.export) == ["Tensor", "equinox", "onnx", "sklearn", "table"]


def test_table_export_is_self_contained_and_contract_driven() -> None:
    import numpy as np
    from bindnordic.models.artifact import load, read_payload

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifold.fnom"
        exported = fno.Export.table(
            {
                "progress": np.asarray([0.0, 0.5, 1.0]),
                "variance": np.asarray([0.0, 1.0]),
            },
            np.arange(12, dtype=float).reshape(3, 2, 2),
            path=path,
            outputs={"state": fno.Tensor.vector(components=2)},
            bounds={"progress": "clamp", "variance": "error"},
        )

        artifact = load(exported)
        assert exported == path.resolve()
        assert artifact.format == "table"
        assert artifact.runtime == "table"
        assert [port.name for port in artifact.inputs] == ["progress", "variance"]
        assert artifact.outputs[0].components == 2
        assert read_payload(path).startswith(b"FNOTAB1\0")
        assert not path.with_suffix(".table").exists()


def test_table_export_validates_axes_shape_and_bounds() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.fnom"
        with pytest.raises(ValueError, match="strictly increasing"):
            fno.Export.table(
                {"x": [0.0, 0.0]},
                np.zeros(2),
                path=path,
                outputs={"y": fno.Tensor.scalar()},
            )
        with pytest.raises(ValueError, match="shape"):
            fno.Export.table(
                {"x": [0.0, 1.0]},
                np.zeros(3),
                path=path,
                outputs={"y": fno.Tensor.scalar()},
            )


def test_verbose_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="boolean"):
        fno.export.onnx(
            b"onnx",
            path="model.fnom",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
            verbose="yes",
        )


def test_scalers_default_to_none_for_every_exporter() -> None:
    for exporter in (
        fno.export.onnx,
        fno.export.sklearn,
        fno.export.equinox,
    ):
        parameters = inspect.signature(exporter).parameters
        assert parameters["x_scaler"].default is None
        assert parameters["y_scaler"].default is None


def test_compiled_rejects_unsupported_estimators_before_writing() -> None:
    from sklearn.gaussian_process import GaussianProcessRegressor

    model = GaussianProcessRegressor().fit([[0.0], [1.0]], [0.0, 1.0])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gpr.fnom"
        with pytest.raises(
            TypeError, match="does not support GaussianProcessRegressor"
        ):
            fno.export.sklearn(
                model,
                path=path,
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                backend="compiled",
            )
        assert not path.exists()


def test_joblib_runtime_is_strict() -> None:
    with pytest.raises(ValueError, match="'sklearn' or 'sklearnex'"):
        fno.export.sklearn(
            _LinearPrediction(),
            path="model.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
            runtime="automatic",
        )


def test_sklearnex_runtime_patches_the_resident_process(mocker) -> None:
    from bindnordic.models.runtime import _activate_joblib_runtime

    patch_sklearn = mocker.Mock()
    module = types.SimpleNamespace(patch_sklearn=patch_sklearn)
    _mock_patch_3 = mocker.patch.dict(sys.modules, {"sklearnex": module})
    try:
        _activate_joblib_runtime("sklearnex")
    finally:
        mocker.stop(_mock_patch_3)
    patch_sklearn.assert_called_once_with(verbose=False)


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_sklearnex_runtime_is_stored_in_fnom() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manifest = fno.export.sklearn(
            _LinearPrediction(),
            path=Path(directory) / "accelerated.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
            runtime="sklearnex",
        )
        metadata = fno._native.read_model_manifest(str(manifest))
    assert metadata["runtime"] == "sklearnex"


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_sklearn_input_format_is_stored_in_fnom() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manifest = fno.export.sklearn(
            _LinearPrediction(),
            path=Path(directory) / "named.fnom",
            inputs={
                "first": fno.Tensor.scalar(),
                "second": fno.Tensor.scalar(),
            },
            outputs={"prediction": fno.Tensor.scalar()},
            input_format="pandas",
        )
        metadata = fno._native.read_model_manifest(str(manifest))
    assert metadata["input_format"] == "pandas"


def test_named_sklearn_input_format_rejects_vector_ports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="require scalar input ports"):
            fno.export.sklearn(
                _LinearPrediction(),
                path=Path(directory) / "named.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"prediction": fno.Tensor.scalar()},
                input_format="pandas",
            )


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_existing_onnx_payload_exports_self_contained_bundle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = fno.export.onnx(
            b"onnx-payload",
            path=root / "reaction-rate.fnom",
            name="reaction-rate",
            inputs={
                "c_tilde": fno.Tensor.scalar(),
                "c_var": fno.Tensor.scalar(),
                "T_tilde": fno.Tensor.scalar(),
            },
            outputs={"omega_c": fno.Tensor.scalar()},
        )
        assert manifest.read_bytes()[:8] == b"FNOMBIN\0"
        assert fno._native.read_model_payload(str(manifest)) == b"onnx-payload"
        assert not (root / "reaction-rate.onnx").exists()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_models_load_exposes_read_only_fnom_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = fno.export.onnx(
            b"onnx-payload",
            path=Path(directory) / "reaction-rate.fnom",
            name="reaction-rate",
            inputs={"c": fno.Tensor.scalar()},
            outputs={"omega_c": fno.Tensor.scalar()},
        )
        artifact = fno.Models.load(path)

    assert artifact.name == "reaction-rate"
    assert artifact.backend == "onnx"
    assert artifact.inputs[0].name == "c"
    assert artifact.outputs[0].name == "omega_c"
    assert artifact.input_format == "numpy"
    assert artifact.payload_size == len(b"onnx-payload")
    assert "FnomArtifact" in dir(fno.Models)


def test_models_load_requires_fnom_extension() -> None:
    with pytest.raises(ValueError, match=r"\.fnom extension"):
        fno.Models.load("model.onnx")


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_path_backed_payload_is_embedded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "large-voting-regressor.onnx"
        source.write_bytes(b"model" * 1024)
        manifest = fno.export.onnx(
            source,
            path=root / "voting.fnom",
            inputs={"features": fno.Tensor.vector(components=3)},
            outputs={"prediction": fno.Tensor.scalar()},
        )
        assert fno._native.read_model_payload(str(manifest)) == source.read_bytes()
        assert not manifest.with_suffix(".onnx").exists()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_verbose_true_displays_export_summary(mocker) -> None:
    display = mocker.patch("foamnordic.export._display_export")
    try:
        with (
            tempfile.TemporaryDirectory() as directory,
        ):
            fno.export.onnx(
                b"onnx-payload",
                path=Path(directory) / "model.fnom",
                inputs={"x": fno.Tensor.scalar()},
                outputs={"y": fno.Tensor.scalar()},
                verbose=True,
            )
    finally:
        mocker.stop(display)
    display.assert_called_once()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_fitted_sklearn_scalers_are_normalized_into_fnom() -> None:
    try:
        import numpy as np
        from sklearn.preprocessing import (
            FunctionTransformer,
            MaxAbsScaler,
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )
    except ImportError:
        pytest.skip("scikit-learn test dependencies are unavailable")

    features = np.asarray([[0.0, -2.0], [2.0, 4.0], [4.0, 10.0]], dtype=np.float64)
    targets = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)
    scalers = (
        (StandardScaler().fit(features), "standard"),
        (MinMaxScaler(feature_range=(-1.0, 1.0), clip=True).fit(features), "minmax"),
        (MaxAbsScaler().fit(features), "maxabs"),
        (RobustScaler().fit(features), "robust"),
        (
            FunctionTransformer(
                func=lambda values: (
                    values * np.asarray([2.0, -4.0]) + np.asarray([3.0, 5.0])
                ),
                inverse_func=lambda values: (
                    (values - np.asarray([3.0, 5.0])) / np.asarray([2.0, -4.0])
                ),
            ).fit(features),
            "function",
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (x_scaler, expected_kind) in enumerate(scalers):
            manifest = fno.export.onnx(
                b"onnx-payload",
                path=root / f"scaled-{index}.fnom",
                inputs={"features": fno.Tensor.vector(components=2)},
                outputs={"prediction": fno.Tensor.scalar()},
                x_scaler=x_scaler,
                y_scaler=StandardScaler().fit(targets),
            )
            metadata = fno._native.read_model_manifest(str(manifest))
            assert metadata["input_scaler"]["kind"] == expected_kind
            assert metadata["output_scaler"]["kind"] == "standard"
            assert len(metadata["input_scaler"]["gain"]) == 2
            assert len(metadata["output_scaler"]["gain"]) == 1


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_nonlinear_function_transformer_is_rejected() -> None:
    try:
        import numpy as np
        from sklearn.preprocessing import FunctionTransformer
    except ImportError:
        pytest.skip("scikit-learn test dependencies are unavailable")
    scaler = FunctionTransformer(np.square).fit(np.asarray([[0.0, 1.0], [1.0, 2.0]]))
    with (
        tempfile.TemporaryDirectory() as directory,
        pytest.raises(ValueError, match="nonlinear"),
    ):
        fno.export.onnx(
            b"onnx-payload",
            path=Path(directory) / "nonlinear.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
            x_scaler=scaler,
        )


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_scaler_feature_mismatch_fails_during_export() -> None:
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pytest.skip("scikit-learn test dependencies are unavailable")
    scaler = StandardScaler().fit(np.asarray([[0.0], [1.0]]))
    with (
        tempfile.TemporaryDirectory() as directory,
        pytest.raises(ValueError, match="feature"),
    ):
        fno.export.onnx(
            b"onnx-payload",
            path=Path(directory) / "mismatch.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
            x_scaler=scaler,
        )


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_joblib_model_is_embedded_and_loadable(mocker) -> None:
    try:
        import joblib
        import numpy as np
    except ImportError:
        pytest.skip("Joblib test dependencies are unavailable")
    from bindnordic.models.runtime import _load_joblib
    from foamnordic.execution.resident import (
        _EmbeddedPayload,
        _joblib_evaluator,
        _payload,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = fno.export.sklearn(
            _LinearPrediction(),
            path=root / "voting.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
        )
        metadata = fno._native.read_model_manifest(str(manifest))
        assert metadata["format"] == "joblib"
        assert metadata["runtime"] == "sklearn"
        assert metadata["bundled"]
        assert isinstance(_payload(manifest, metadata), BytesIO)
        payload = BytesIO(fno._native.read_model_payload(str(manifest)))
        loaded = joblib.load(payload)
        np.testing.assert_allclose(loaded.weights, [2.0, 1.0])
        values = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)
        evaluator = _joblib_evaluator(payload, (("prediction", 1),))
        result = evaluator(memoryview(values), 2, 2, "float64", 0, 0.0)
        np.testing.assert_allclose(result, [[5.0], [8.0]])

        large_model = _LinearPrediction()
        large_model.padding = np.zeros(4 * 1024 * 1024, dtype=np.float64)
        large_manifest = fno.export.sklearn(
            large_model,
            path=root / "large.fnom",
            inputs={"features": fno.Tensor.vector(components=2)},
            outputs={"prediction": fno.Tensor.scalar()},
        )
        large_metadata = fno._native.read_model_manifest(str(large_manifest))
        embedded = _payload(large_manifest, large_metadata)
        assert isinstance(embedded, _EmbeddedPayload)
        mapped, storage, mode = _load_joblib(embedded, joblib)
        assert isinstance(mapped.padding, np.memmap)
        assert mode == "direct"
        assert embedded.offset % 64 == 0
        assert storage is None
        assert Path(mapped.padding.filename).samefile(large_manifest)
        assert mapped.padding.offset >= embedded.offset
        _mock_patch_5 = mocker.patch(
            "bindnordic.models.runtime.inspect.signature",
            return_value=inspect.Signature(),
        )
        try:
            staged, staged_storage, staged_mode = _load_joblib(embedded, joblib)
        finally:
            mocker.stop(_mock_patch_5)
        assert staged_mode == "staged"
        assert staged_storage is not None
        assert isinstance(staged.padding, np.memmap)


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_equinox_model_exports_tree_metadata_and_evaluates() -> None:
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError:
        pytest.skip("Equinox test dependencies are unavailable")
    from foamnordic.execution.resident import _equinox_evaluator

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(42))
        manifest = fno.export.equinox(
            model,
            path=root / "closure.fnom",
            inputs={"features": fno.Tensor.vector(components=2, dtype="float32")},
            outputs={"prediction": fno.Tensor.scalar(dtype="float32")},
        )
        metadata = fno._native.read_model_manifest(str(manifest))
        assert metadata["format"] == "equinox"
        payload = BytesIO(fno._native.read_model_payload(str(manifest)))
        evaluator = _equinox_evaluator(payload, (("prediction", 1),), "float32")
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = evaluator(memoryview(values), 2, 2, "float32", 0, 0.0)
        expected = jax.vmap(model)(jnp.asarray(values))
        np.testing.assert_allclose(result, expected, rtol=1e-6)
