#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Value>
foamnordic::fjord::MutableTensorView matrix_view(
    std::vector<Value>& values,
    std::uint64_t rows,
    std::uint64_t features) {
    const auto element = sizeof(Value) == sizeof(float)
                             ? foamnordic::fjord::Element::float32
                             : foamnordic::fjord::Element::float64;
    return {
        "features",
        element,
        {rows, features},
        foamnordic::fjord::as_writable_bytes(std::span(values)),
        1,
        0.1,
    };
}

template <class Value>
foamnordic::fjord::TensorView readonly_matrix_view(
    const std::vector<Value>& values,
    std::uint64_t rows,
    std::uint64_t features) {
    const auto element = sizeof(Value) == sizeof(float)
                             ? foamnordic::fjord::Element::float32
                             : foamnordic::fjord::Element::float64;
    return {
        "features",
        element,
        {rows, features},
        foamnordic::fjord::as_bytes(std::span(values)),
        1,
        0.1,
    };
}

void require_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(std::abs(actual[index] - expected[index]) < 1.0e-12, message);
    }
}

void test_standard_scaler() {
    auto scaler = foamnordic::closure::AffineScaler::standard(
        {10.0, 100.0},
        {2.0, 20.0});
    std::vector<double> values{10.0, 120.0, 14.0, 80.0};
    scaler.transform(matrix_view(values, 2, 2));
    require_close(values, {0.0, 1.0, 2.0, -1.0}, "StandardScaler transform failed.");
    scaler.inverse_transform(matrix_view(values, 2, 2));
    require_close(values, {10.0, 120.0, 14.0, 80.0}, "StandardScaler inverse failed.");
}

void test_minmax_scaler_with_clipping() {
    auto scaler = foamnordic::closure::AffineScaler::minmax(
        {0.5, 0.25},
        {-1.0, 0.0},
        0.0,
        1.0);
    std::vector<double> values{0.0, 2.0, 10.0, 8.0};
    scaler.transform(matrix_view(values, 2, 2));
    require_close(values, {0.0, 0.5, 1.0, 1.0}, "MinMaxScaler transform failed.");
}

void test_robust_scaler_float32() {
    auto scaler = foamnordic::closure::AffineScaler::robust(
        {5.0, 20.0},
        {2.0, 10.0});
    std::vector<float> values{7.0F, 10.0F};
    scaler.transform(matrix_view(values, 1, 2));
    require(
        values == std::vector<float>{1.0F, -1.0F},
        "RobustScaler float32 transform failed.");
}

void test_maxabs_scaler_through_common_interface() {
    std::unique_ptr<foamnordic::closure::Scaler> scaler =
        std::make_unique<foamnordic::closure::AffineScaler>(
            foamnordic::closure::AffineScaler::maxabs({2.0, 10.0}));
    std::vector<double> values{-2.0, 5.0, 1.0, -10.0};
    scaler->transform(matrix_view(values, 2, 2));
    require_close(
        values,
        {-1.0, 0.5, 0.5, -1.0},
        "MaxAbsScaler transform failed.");
    scaler->inverse_transform(matrix_view(values, 2, 2));
    require_close(
        values,
        {-2.0, 5.0, 1.0, -10.0},
        "MaxAbsScaler inverse failed.");
}

foamnordic::closure::ClosureContract contract() {
    return {
        "combustion",
        {
            {"c_tilde", foamnordic::fjord::Element::float64, 1},
            {"c_var", foamnordic::fjord::Element::float64, 1},
            {"T_tilde", foamnordic::fjord::Element::float64, 1},
        },
        {{"omega_c", foamnordic::fjord::Element::float64, 1}},
    };
}

void test_three_artifact_formats() {
    for (const auto format : {
             foamnordic::closure::ModelFormat::joblib,
             foamnordic::closure::ModelFormat::onnx,
         }) {
        foamnordic::closure::ModelArtifact artifact{
            1,
            format,
            "artifact/model.bin",
            contract(),
            {},
            foamnordic::closure::AffineScaler::standard(
                {0.0, 0.0, 300.0},
                {1.0, 1.0, 500.0}),
            foamnordic::closure::AffineScaler::robust({0.0}, {10.0}),
        };
        artifact.validate();
    }

    foamnordic::closure::ModelArtifact equinox{
        1,
        foamnordic::closure::ModelFormat::equinox,
        "artifact/weights.eqx",
        contract(),
        {
            {"layers[0].weight", foamnordic::fjord::Element::float64, {3, 4}, 0, 96},
            {"layers[0].bias", foamnordic::fjord::Element::float64, {4}, 96, 32},
        },
        std::nullopt,
        std::nullopt,
    };
    equinox.validate();
}

void test_scaler_feature_mismatch_is_rejected() {
    foamnordic::closure::ModelArtifact artifact{
        1,
        foamnordic::closure::ModelFormat::onnx,
        "artifact/model.onnx",
        contract(),
        {},
        foamnordic::closure::AffineScaler::standard({0.0}, {1.0}),
        std::nullopt,
    };
    bool rejected = false;
    try {
        artifact.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Model artifact accepted a mismatched input scaler.");
}

foamnordic::fjord::Tensor scalar_tensor(
    std::string name,
    const std::vector<double>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        foamnordic::fjord::Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        9,
        0.25,
    };
}

class InspectingPackedKernel final : public foamnordic::closure::PackedModelKernel {
public:
    foamnordic::fjord::Tensor evaluate(
        foamnordic::fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time) override {
        require(exchange_index == 9 && physical_time == 0.25, "Packed exchange metadata failed.");
        std::vector<double> values(features.bytes.size() / sizeof(double));
        std::memcpy(values.data(), features.bytes.data(), features.bytes.size());
        require_close(values, {0.5, 2.0, 2.0}, "Native input packing or scaling failed.");
        auto output = scalar_tensor("foamnordic.predictions", {3.0});
        output.shape = {1, 1};
        return output;
    }
};

void test_artifact_kernel_packs_scales_and_unpacks() {
    InspectingPackedKernel packed_kernel;
    foamnordic::closure::ModelArtifact artifact{
        1,
        foamnordic::closure::ModelFormat::onnx,
        "artifact/model.onnx",
        contract(),
        {},
        foamnordic::closure::AffineScaler::standard(
            {0.0, 0.0, 300.0},
            {1.0, 0.1, 100.0}),
        foamnordic::closure::AffineScaler::standard({10.0}, {2.0}),
    };
    foamnordic::closure::ArtifactModelKernel kernel(std::move(artifact), packed_kernel);
    foamnordic::closure::TensorMap inputs;
    inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.5}));
    inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.2}));
    inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 500.0}));

    const auto outputs = kernel.evaluate(inputs, {1}, 9, 0.25);
    const auto& omega = outputs.at("omega_c");
    std::vector<double> values(1);
    std::memcpy(values.data(), omega.bytes.data(), omega.bytes.size());
    require_close(values, {16.0}, "Native output inverse scaling or unpacking failed.");
}

void test_dense_affine_kernel_float64() {
    foamnordic::closure::DenseAffineKernel kernel(
        3,
        2,
        {
            1.0, 2.0, 3.0,
            -1.0, 0.5, 2.0,
        },
        {0.5, -2.0});
    std::vector<double> values{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const auto output = kernel.evaluate(readonly_matrix_view(values, 2, 3), 1, 0.1);
    std::vector<double> actual(output.bytes.size() / sizeof(double));
    std::memcpy(actual.data(), output.bytes.data(), output.bytes.size());
    require_close(
        actual,
        {14.5, 4.0, 32.5, 8.5},
        "Dense affine float64 evaluation failed.");
}

void test_dense_affine_kernel_float32() {
    foamnordic::closure::DenseAffineKernel kernel(
        2,
        1,
        {2.0, -0.5},
        {1.0});
    std::vector<float> values{3.0F, 4.0F, -1.0F, 2.0F};
    const auto output = kernel.evaluate(readonly_matrix_view(values, 2, 2), 1, 0.1);
    std::vector<float> actual(output.bytes.size() / sizeof(float));
    std::memcpy(actual.data(), output.bytes.data(), output.bytes.size());
    require(
        actual == std::vector<float>{5.0F, -2.0F},
        "Dense affine float32 evaluation failed.");
}

void test_manifest_round_trip_and_corruption_rejection() {
    foamnordic::closure::ModelArtifact original{
        1,
        foamnordic::closure::ModelFormat::equinox,
        "weights/combustion.eqx",
        contract(),
        {
            {"layers[0].weight", foamnordic::fjord::Element::float64, {3, 2}, 0, 48},
            {"layers[0].bias", foamnordic::fjord::Element::float64, {2}, 48, 16},
        },
        foamnordic::closure::AffineScaler::standard(
            {0.0, 0.0, 300.0},
            {1.0, 1.0, 500.0}),
        foamnordic::closure::AffineScaler::minmax(
            {0.25},
            {-1.0},
            0.0,
            1.0),
    };
    const auto encoded = foamnordic::closure::encode_manifest(original);
    const auto decoded = foamnordic::closure::decode_manifest(encoded);
    const auto manifest_path = std::filesystem::temp_directory_path()
                               / ("foamnordic-manifest-"
                                  + std::to_string(std::random_device{}()) + ".fnom");
    foamnordic::closure::write_manifest(manifest_path, original);
    const auto loaded = foamnordic::closure::read_manifest(manifest_path);
    std::error_code remove_error;
    std::filesystem::remove(manifest_path, remove_error);
    require(!remove_error, "Temporary manifest cleanup failed.");
    require(
        loaded.contract.name == original.contract.name
            && loaded.artifact_path == original.artifact_path,
        "Manifest file loader did not round-trip model identity.");
    require(decoded.schema_version == 1, "Manifest schema did not round-trip.");
    require(
        decoded.format == foamnordic::closure::ModelFormat::equinox
            && decoded.artifact_path == original.artifact_path,
        "Manifest model identity did not round-trip.");
    require(
        decoded.contract.name == original.contract.name
            && decoded.contract.inputs.size() == 3
            && decoded.contract.outputs.size() == 1,
        "Manifest closure contract did not round-trip.");
    require(
        decoded.tree_leaves.size() == 2
            && decoded.tree_leaves[1].byte_offset == 48
            && decoded.tree_leaves[1].byte_count == 16,
        "Manifest Equinox leaves did not round-trip.");
    require(
        decoded.input_scaler.has_value()
            && decoded.input_scaler->gain() == original.input_scaler->gain()
            && decoded.input_scaler->bias() == original.input_scaler->bias(),
        "Manifest input scaler did not round-trip.");
    require(
        decoded.output_scaler.has_value()
            && decoded.output_scaler->clip_lower() == 0.0
            && decoded.output_scaler->clip_upper() == 1.0,
        "Manifest output clipping did not round-trip.");

    auto corrupted = encoded;
    corrupted.front() = std::byte{0};
    bool bad_magic_rejected = false;
    try {
        static_cast<void>(foamnordic::closure::decode_manifest(corrupted));
    } catch (const std::invalid_argument&) {
        bad_magic_rejected = true;
    }
    require(bad_magic_rejected, "Manifest accepted corrupted magic.");

    auto trailing = encoded;
    trailing.push_back(std::byte{0});
    bool trailing_rejected = false;
    try {
        static_cast<void>(foamnordic::closure::decode_manifest(trailing));
    } catch (const std::invalid_argument&) {
        trailing_rejected = true;
    }
    require(trailing_rejected, "Manifest accepted trailing data.");
}

}  // namespace

int main() {
    test_standard_scaler();
    test_minmax_scaler_with_clipping();
    test_robust_scaler_float32();
    test_maxabs_scaler_through_common_interface();
    test_three_artifact_formats();
    test_scaler_feature_mismatch_is_rejected();
    test_artifact_kernel_packs_scales_and_unpacks();
    test_dense_affine_kernel_float64();
    test_dense_affine_kernel_float32();
    test_manifest_round_trip_and_corruption_rejection();
}
