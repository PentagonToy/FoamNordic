#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/smedja.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Operation>
void require_throws(Operation&& operation, const std::string& message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(stream.good(), "Could not open temporary bundle.");
    const auto count = stream.tellg();
    require(count >= 0, "Temporary bundle size was invalid.");
    stream.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(count));
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(stream.good(), "Could not read temporary bundle.");
    return bytes;
}

void write_bytes(
    const std::filesystem::path& path,
    const std::vector<std::byte>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good(), "Could not create malformed temporary bundle.");
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(stream.good(), "Could not write malformed temporary bundle.");
}

void write_u64(std::vector<std::byte>& bytes, std::size_t offset, std::uint64_t value) {
    require(offset + sizeof(value) <= bytes.size(), "Bundle test offset was invalid.");
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[offset++] = static_cast<std::byte>((value >> shift) & 0xffU);
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
    auto scaler = foamnordic::inference::AffineScaler::standard(
        {10.0, 100.0},
        {2.0, 20.0});
    std::vector<double> values{10.0, 120.0, 14.0, 80.0};
    scaler.transform(matrix_view(values, 2, 2));
    require_close(values, {0.0, 1.0, 2.0, -1.0}, "StandardScaler transform failed.");
    scaler.inverse_transform(matrix_view(values, 2, 2));
    require_close(values, {10.0, 120.0, 14.0, 80.0}, "StandardScaler inverse failed.");
}

void test_minmax_scaler_with_clipping() {
    auto scaler = foamnordic::inference::AffineScaler::minmax(
        {0.5, 0.25},
        {-1.0, 0.0},
        0.0,
        1.0);
    std::vector<double> values{0.0, 2.0, 10.0, 8.0};
    scaler.transform(matrix_view(values, 2, 2));
    require_close(values, {0.0, 0.5, 1.0, 1.0}, "MinMaxScaler transform failed.");
}

void test_robust_scaler_float32() {
    auto scaler = foamnordic::inference::AffineScaler::robust(
        {5.0, 20.0},
        {2.0, 10.0});
    std::vector<float> values{7.0F, 10.0F};
    scaler.transform(matrix_view(values, 1, 2));
    require(
        values == std::vector<float>{1.0F, -1.0F},
        "RobustScaler float32 transform failed.");
}

void test_maxabs_scaler_through_common_interface() {
    std::unique_ptr<foamnordic::inference::Scaler> scaler =
        std::make_unique<foamnordic::inference::AffineScaler>(
            foamnordic::inference::AffineScaler::maxabs({2.0, 10.0}));
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

foamnordic::inference::ProgramContract contract() {
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
             foamnordic::inference::ModelFormat::joblib,
             foamnordic::inference::ModelFormat::onnx,
         }) {
        foamnordic::inference::ModelArtifact artifact{
            1,
            format,
            "artifact/model.bin",
            contract(),
            {},
            foamnordic::inference::AffineScaler::standard(
                {0.0, 0.0, 300.0},
                {1.0, 1.0, 500.0}),
            foamnordic::inference::AffineScaler::robust({0.0}, {10.0}),
        };
        artifact.validate();
    }

    foamnordic::inference::ModelArtifact equinox{
        1,
        foamnordic::inference::ModelFormat::equinox,
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
    foamnordic::inference::ModelArtifact artifact{
        1,
        foamnordic::inference::ModelFormat::onnx,
        "artifact/model.onnx",
        contract(),
        {},
        foamnordic::inference::AffineScaler::standard({0.0}, {1.0}),
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

foamnordic::fjord::Tensor float_tensor(
    std::string name,
    const std::vector<float>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        foamnordic::fjord::Element::float32,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        9,
        0.25,
    };
}

class InspectingPackedKernel final : public foamnordic::inference::PackedModelKernel {
public:
    foamnordic::fjord::Tensor evaluate(
        foamnordic::fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t /*rank*/) override {
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
    foamnordic::inference::ModelArtifact artifact{
        1,
        foamnordic::inference::ModelFormat::onnx,
        "artifact/model.onnx",
        contract(),
        {},
        foamnordic::inference::AffineScaler::standard(
            {0.0, 0.0, 300.0},
            {1.0, 0.1, 100.0}),
        foamnordic::inference::AffineScaler::standard({10.0}, {2.0}),
    };
    foamnordic::inference::ArtifactModelKernel kernel(std::move(artifact), packed_kernel);
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.5}));
    inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.2}));
    inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 500.0}));

    const auto outputs = kernel.evaluate(inputs, {1}, 9, 0.25);
    const auto& omega = outputs.at("omega_c");
    std::vector<double> values(1);
    std::memcpy(values.data(), omega.bytes.data(), omega.bytes.size());
    require_close(values, {16.0}, "Native output inverse scaling or unpacking failed.");
}

void test_smedja_rebinds_dynamic_input_storage() {
    foamnordic::inference::Smedja smedja(contract());

    foamnordic::inference::TensorMap first;
    first.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
    first.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
    first.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
    const auto first_packed = smedja.pack(first, {1}, 9, 0.25);
    std::vector<double> first_values(3);
    std::memcpy(
        first_values.data(),
        first_packed.bytes.data(),
        first_packed.bytes.size());
    require_close(
        first_values,
        {0.2, 0.02, 400.0},
        "Smedja failed its first dynamic input binding.");

    foamnordic::inference::TensorMap second;
    second.emplace("c_tilde", scalar_tensor("c_tilde", {1.0, 2.0, 3.0}));
    second.emplace("c_var", scalar_tensor("c_var", {0.1, 0.2, 0.3}));
    second.emplace("T_tilde", scalar_tensor("T_tilde", {500.0, 600.0, 700.0}));
    const auto second_packed = smedja.pack(second, {2, 0}, 10, 0.5);
    std::vector<double> second_values(6);
    std::memcpy(
        second_values.data(),
        second_packed.bytes.data(),
        second_packed.bytes.size());
    require_close(
        second_values,
        {3.0, 0.3, 700.0, 1.0, 0.1, 500.0},
        "Smedja retained stale input storage or shape metadata.");
}

void test_smedja_moves_a_single_output_staging_buffer() {
    foamnordic::inference::Smedja smedja(contract());
    auto packed = scalar_tensor("foamnordic.predictions", {1.0, 2.0});
    packed.shape = {2, 1};
    const auto* storage = packed.bytes.data();

    auto outputs = smedja.unpack(std::move(packed));
    const auto& output = outputs.at("omega_c");
    require(
        output.bytes.data() == storage
            && output.shape == std::vector<std::uint64_t>{2},
        "Smedja copied a compatible single-output staging buffer.");
}

void test_smedja_reuses_multi_output_staging_for_widest_field() {
    const foamnordic::inference::ProgramContract multi_output_contract{
        "multi-output",
        {{"input", foamnordic::fjord::Element::float64, 1}},
        {
            {"scalar", foamnordic::fjord::Element::float64, 1},
            {"vector", foamnordic::fjord::Element::float64, 2},
        },
    };
    foamnordic::inference::Smedja smedja(multi_output_contract);
    auto packed = scalar_tensor(
        "foamnordic.predictions",
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    packed.shape = {2, 3};
    const auto* storage = packed.bytes.data();

    const auto outputs = smedja.unpack(std::move(packed));
    const auto& scalar = outputs.at("scalar");
    const auto& vector = outputs.at("vector");
    std::vector<double> scalar_values(2);
    std::vector<double> vector_values(4);
    std::memcpy(
        scalar_values.data(), scalar.bytes.data(), scalar.bytes.size());
    std::memcpy(
        vector_values.data(), vector.bytes.data(), vector.bytes.size());

    require_close(
        scalar_values,
        {1.0, 4.0},
        "Smedja changed a scalar output while compacting staging.");
    require_close(
        vector_values,
        {2.0, 3.0, 5.0, 6.0},
        "Smedja changed a vector output while compacting staging.");
    require(
        vector.bytes.data() == storage,
        "Smedja did not transfer multi-output staging to its widest field.");
}

void test_smedja_reuses_and_bounds_workspace_capacity() {
    foamnordic::inference::Smedja smedja(contract());
    foamnordic::inference::SmedjaWorkspace workspace;
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c_tilde", scalar_tensor("c_tilde", std::vector<double>(64, 0.1)));
    inputs.emplace("c_var", scalar_tensor("c_var", std::vector<double>(64, 0.01)));
    inputs.emplace("T_tilde", scalar_tensor("T_tilde", std::vector<double>(64, 300.0)));

    auto& first = smedja.pack_into(workspace, inputs, {0, 1, 2, 3}, 10, 0.1);
    const auto* storage = first.bytes.data();
    auto& second = smedja.pack_into(workspace, inputs, {4, 5, 6, 7}, 11, 0.2);
    require(
        second.bytes.data() == storage,
        "Smedja did not reuse a compatible workspace allocation.");

    std::vector<std::uint64_t> all_cells(64);
    for (std::uint64_t cell = 0; cell < all_cells.size(); ++cell) {
        all_cells[cell] = cell;
    }
    static_cast<void>(smedja.pack_into(workspace, inputs, all_cells, 12, 0.3));
    const auto large_capacity = workspace.retained_bytes();
    static_cast<void>(smedja.pack_into(workspace, inputs, {0}, 13, 0.4));
    require(
        workspace.retained_bytes() < large_capacity,
        "Smedja retained an oversized dynamic-mesh workspace.");
}

void test_smedja_packs_contiguous_and_sparse_cell_ranges() {
    const foamnordic::inference::ProgramContract single_input_contract{
        "single",
        {{"value", foamnordic::fjord::Element::float64, 1}},
        {{"result", foamnordic::fjord::Element::float64, 1}},
    };
    foamnordic::inference::Smedja smedja(single_input_contract);
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("value", scalar_tensor("value", {10.0, 20.0, 30.0, 40.0, 50.0}));

    const auto contiguous = smedja.pack(inputs, {1, 2, 3}, 20, 0.2);
    std::vector<double> contiguous_values(3);
    std::memcpy(
        contiguous_values.data(), contiguous.bytes.data(), contiguous.bytes.size());
    require_close(
        contiguous_values,
        {20.0, 30.0, 40.0},
        "Smedja contiguous-range packing changed field values.");

    const auto sparse = smedja.pack(inputs, {0, 2, 4}, 21, 0.3);
    std::vector<double> sparse_values(3);
    std::memcpy(sparse_values.data(), sparse.bytes.data(), sparse.bytes.size());
    require_close(
        sparse_values,
        {10.0, 30.0, 50.0},
        "Smedja sparse gather packing changed field values.");
}

void test_smedja_fuses_float_input_conversion_with_packing() {
    foamnordic::inference::Smedja smedja(contract());
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c_tilde", float_tensor("c_tilde", {0.25F, 0.5F}));
    inputs.emplace("c_var", float_tensor("c_var", {0.01F, 0.02F}));
    inputs.emplace("T_tilde", float_tensor("T_tilde", {300.0F, 400.0F}));

    const auto packed = smedja.pack(inputs, {1, 0}, 22, 0.4);
    std::vector<double> values(6);
    std::memcpy(values.data(), packed.bytes.data(), packed.bytes.size());
    require_close(
        values,
        {0.5, static_cast<double>(0.02F), 400.0,
         0.25, static_cast<double>(0.01F), 300.0},
        "Smedja did not fuse float32-to-float64 conversion with packing.");
}

class ConcurrentProbeKernel final : public foamnordic::inference::PackedModelKernel {
public:
    foamnordic::fjord::Tensor evaluate(
        foamnordic::fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t /*rank*/) override {
        const auto active = active_.fetch_add(1) + 1;
        auto peak = peak_.load();
        while (active > peak && !peak_.compare_exchange_weak(peak, active)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        active_.fetch_sub(1);
        auto output = scalar_tensor(
            "foamnordic.predictions",
            std::vector<double>(features.shape.front(), 1.0));
        output.shape = {features.shape.front(), 1};
        output.time_index = exchange_index;
        output.physical_time = physical_time;
        return output;
    }

    [[nodiscard]] std::uint32_t peak() const noexcept { return peak_.load(); }

private:
    std::atomic<std::uint32_t> active_{0};
    std::atomic<std::uint32_t> peak_{0};
};

void test_artifact_kernel_owns_a_narrow_backend_lock() {
    ConcurrentProbeKernel packed_kernel;
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            1,
            foamnordic::inference::ModelFormat::onnx,
            "artifact/model.onnx",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
        },
        packed_kernel);
    require(
        kernel.owns_backend_synchronization(),
        "Artifact kernel did not advertise its backend synchronization boundary.");

    const auto invoke = [&](std::uint64_t exchange_index) {
        foamnordic::inference::TensorMap inputs;
        inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
        inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
        inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
        static_cast<void>(kernel.evaluate(inputs, {0, 1}, exchange_index, 0.25));
    };

    std::thread first([&] { invoke(10); });
    std::thread second([&] { invoke(11); });
    first.join();
    second.join();
    require(
        packed_kernel.peak() == 1,
        "Artifact kernel allowed concurrent access to a serialized backend.");
}

void test_dense_affine_kernel_float64() {
    foamnordic::inference::DenseAffineKernel kernel(
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
    foamnordic::inference::DenseAffineKernel kernel(
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
    foamnordic::inference::ModelArtifact original{
        1,
        foamnordic::inference::ModelFormat::equinox,
        "weights/combustion.eqx",
        contract(),
        {
            {"layers[0].weight", foamnordic::fjord::Element::float64, {3, 2}, 0, 48},
            {"layers[0].bias", foamnordic::fjord::Element::float64, {2}, 48, 16},
        },
        foamnordic::inference::AffineScaler::standard(
            {0.0, 0.0, 300.0},
            {1.0, 1.0, 500.0}),
        foamnordic::inference::AffineScaler::minmax(
            {0.25},
            {-1.0},
            0.0,
            1.0),
    };
    const auto encoded = foamnordic::inference::encode_manifest(original);
    const auto decoded = foamnordic::inference::decode_manifest(encoded);
    const auto manifest_path = std::filesystem::temp_directory_path()
                               / ("foamnordic-manifest-"
                                  + std::to_string(std::random_device{}()) + ".fnom");
    foamnordic::inference::write_manifest(manifest_path, original);
    const auto loaded = foamnordic::inference::read_manifest(manifest_path);
    std::error_code remove_error;
    std::filesystem::remove(manifest_path, remove_error);
    require(!remove_error, "Temporary manifest cleanup failed.");
    require(
        loaded.contract.name == original.contract.name
            && loaded.artifact_path == original.artifact_path,
        "Manifest file loader did not round-trip model identity.");
    require(decoded.schema_version == 1, "Manifest schema did not round-trip.");
    require(
        decoded.format == foamnordic::inference::ModelFormat::equinox
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

    foamnordic::inference::ModelArtifact accelerated{
        2,
        foamnordic::inference::ModelFormat::joblib,
        "models/closure.joblib",
        contract(),
        {},
        std::nullopt,
        std::nullopt,
        "sklearnex",
    };
    const auto accelerated_round_trip = foamnordic::inference::decode_manifest(
        foamnordic::inference::encode_manifest(accelerated));
    require(
        accelerated_round_trip.schema_version == 2
            && accelerated_round_trip.runtime == "sklearnex",
        "Manifest did not round-trip the Joblib execution runtime.");

    foamnordic::inference::ModelArtifact compiled{
        2,
        foamnordic::inference::ModelFormat::compiled,
        "models/closure.cpp",
        contract(),
        {},
        std::nullopt,
        std::nullopt,
        "cpp-v1",
    };
    const auto compiled_round_trip = foamnordic::inference::decode_manifest(
        foamnordic::inference::encode_manifest(compiled));
    require(
        compiled_round_trip.format
                == foamnordic::inference::ModelFormat::compiled
            && compiled_round_trip.runtime == "cpp-v1",
        "Manifest did not round-trip the compiled execution runtime.");

    auto corrupted = encoded;
    corrupted.front() = std::byte{0};
    bool bad_magic_rejected = false;
    try {
        static_cast<void>(foamnordic::inference::decode_manifest(corrupted));
    } catch (const std::invalid_argument&) {
        bad_magic_rejected = true;
    }
    require(bad_magic_rejected, "Manifest accepted corrupted magic.");

    auto trailing = encoded;
    trailing.push_back(std::byte{0});
    bool trailing_rejected = false;
    try {
        static_cast<void>(foamnordic::inference::decode_manifest(trailing));
    } catch (const std::invalid_argument&) {
        trailing_rejected = true;
    }
    require(trailing_rejected, "Manifest accepted trailing data.");
}

void test_self_contained_bundle_round_trip() {
    const auto nonce = std::to_string(std::random_device{}());
    const auto root = std::filesystem::temp_directory_path();
    const auto payload_path = root / ("foamnordic-payload-" + nonce + ".bin");
    const auto bundle_path = root / ("foamnordic-bundle-" + nonce + ".fnom");
    std::vector<std::byte> expected(8 * 1024 * 1024 + 17);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::byte>(index % 251);
    }
    {
        std::ofstream payload(payload_path, std::ios::binary);
        payload.write(
            reinterpret_cast<const char*>(expected.data()), expected.size());
    }
    foamnordic::inference::ModelArtifact artifact{
        2,
        foamnordic::inference::ModelFormat::joblib,
        payload_path.filename().string(),
        contract(),
        {},
        std::nullopt,
        std::nullopt,
        "sklearn",
    };
    foamnordic::inference::write_bundle(bundle_path, artifact, payload_path);
    require(
        foamnordic::inference::is_bundle(bundle_path),
        "Self-contained FNOM bundle was not detected.");
    const auto loaded = foamnordic::inference::read_manifest(bundle_path);
    require(
        loaded.contract.name == artifact.contract.name
            && loaded.artifact_path == artifact.artifact_path,
        "Bundled manifest metadata did not round-trip.");
    require(
        foamnordic::inference::read_bundle_payload(bundle_path) == expected,
        "Embedded FNOM payload did not round-trip exactly.");
    const auto region = foamnordic::inference::bundle_payload_region(bundle_path);
    require(
        region.offset % 64 == 0 && region.size == expected.size(),
        "FNOM payload region is not aligned for direct memory mapping.");
    const auto extracted_path = root / ("foamnordic-extracted-" + nonce + ".bin");
    foamnordic::inference::extract_bundle_payload(bundle_path, extracted_path);
    require(
        read_bytes(extracted_path) == expected,
        "Streamed FNOM payload did not round-trip exactly.");

    const auto complete = read_bytes(bundle_path);
    const auto malformed_path = root / ("foamnordic-malformed-" + nonce + ".fnom");
    auto malformed = complete;
    malformed.resize(31);
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] { static_cast<void>(foamnordic::inference::read_manifest(malformed_path)); },
        "FNOM bundle accepted a truncated header.");

    malformed = complete;
    malformed.pop_back();
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] {
            static_cast<void>(foamnordic::inference::read_bundle_payload(malformed_path));
        },
        "FNOM bundle accepted a truncated payload.");

    malformed = complete;
    write_u64(malformed, 8, 0);
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] { static_cast<void>(foamnordic::inference::read_manifest(malformed_path)); },
        "FNOM bundle accepted a zero manifest length.");

    malformed = complete;
    write_u64(malformed, 24, expected.size() + 1);
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] {
            static_cast<void>(foamnordic::inference::read_bundle_payload(malformed_path));
        },
        "FNOM bundle accepted an incorrect payload length.");

    malformed = complete;
    write_u64(malformed, 16, 32);
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] {
            static_cast<void>(foamnordic::inference::bundle_payload_region(
                malformed_path));
        },
        "FNOM bundle accepted an overlapping payload region.");

    malformed = complete;
    malformed.push_back(std::byte{0});
    write_bytes(malformed_path, malformed);
    require_throws(
        [&] {
            static_cast<void>(foamnordic::inference::read_bundle_payload(malformed_path));
        },
        "FNOM bundle accepted trailing bytes.");
    require_throws(
        [&] {
            foamnordic::inference::extract_bundle_payload(
                malformed_path, extracted_path);
        },
        "FNOM extraction accepted trailing bytes.");

    const auto legacy_path = root / ("foamnordic-legacy-" + nonce + ".fnom");
    const auto encoded_manifest = foamnordic::inference::encode_manifest(artifact);
    std::vector<std::byte> legacy(24);
    constexpr char legacy_magic[] = "FNOBND1";
    for (std::size_t index = 0; index < 7; ++index) {
        legacy[index] = static_cast<std::byte>(legacy_magic[index]);
    }
    write_u64(legacy, 8, encoded_manifest.size());
    write_u64(legacy, 16, expected.size());
    legacy.insert(legacy.end(), encoded_manifest.begin(), encoded_manifest.end());
    legacy.insert(legacy.end(), expected.begin(), expected.end());
    write_bytes(legacy_path, legacy);
    require(
        foamnordic::inference::read_bundle_payload(legacy_path) == expected,
        "Legacy FNOBND1 payload compatibility failed.");

    std::error_code ignored;
    std::filesystem::remove(payload_path, ignored);
    std::filesystem::remove(bundle_path, ignored);
    std::filesystem::remove(malformed_path, ignored);
    std::filesystem::remove(extracted_path, ignored);
    std::filesystem::remove(legacy_path, ignored);
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
    test_smedja_rebinds_dynamic_input_storage();
    test_smedja_moves_a_single_output_staging_buffer();
    test_smedja_reuses_multi_output_staging_for_widest_field();
    test_smedja_reuses_and_bounds_workspace_capacity();
    test_smedja_packs_contiguous_and_sparse_cell_ranges();
    test_smedja_fuses_float_input_conversion_with_packing();
    test_artifact_kernel_owns_a_narrow_backend_lock();
    test_dense_affine_kernel_float64();
    test_dense_affine_kernel_float32();
    test_manifest_round_trip_and_corruption_rejection();
    test_self_contained_bundle_round_trip();
}
