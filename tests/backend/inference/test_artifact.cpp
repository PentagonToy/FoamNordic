// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * FoamNordic
 */
// clang-format on
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/execution.hpp"
#include "foamnordic/backend/inference/load.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/smedja.hpp"
#include "foamnordic/backend/connectors/connector.hpp"
#include "foamnordic/backend/connectors/registry.hpp"

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

template<class Value>
void append_value(std::vector<std::byte>& bytes, Value value) {
    const auto source = std::as_bytes(std::span(&value, 1));
    bytes.insert(bytes.end(), source.begin(), source.end());
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
    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t /*rank*/) override {
        require(exchange_index == 9 && physical_time == 0.25, "Packed exchange metadata failed.");
        std::vector<double> values(features.bytes.size() / sizeof(double));
        std::memcpy(values.data(), features.bytes.data(), features.bytes.size());
        require_close(values, {0.5, 2.0, 2.0}, "Native input packing or scaling failed.");
        std::vector<std::byte> bytes(sizeof(double));
        const double prediction = 3.0;
        std::memcpy(bytes.data(), &prediction, sizeof(double));
        return {
            "bindnordic.predictions",
            bindnordic::Element::float64,
            {1, 1},
            std::move(bytes)};
    }
};

void test_artifact_kernel_packs_scales_and_unpacks() {
    InspectingPackedKernel packed_kernel;
    foamnordic::inference::ModelArtifact artifact{
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
    auto packed = scalar_tensor("bindnordic.predictions", {1.0, 2.0});
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
        "bindnordic.predictions",
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

    static_cast<void>(smedja.pack_into(workspace, inputs, {}, 14, 0.5));
    require(
        workspace.retained_bytes() == 0,
        "Smedja retained a large workspace for an empty rank partition.");
    static_cast<void>(smedja.pack_into(workspace, inputs, {0}, 15, 0.6));
    require(
        workspace.retained_bytes() > 0,
        "Smedja did not reacquire storage after an empty rank partition.");
    const auto& regrown = smedja.pack_into(
        workspace, inputs, all_cells, 16, 0.7);
    require(
        regrown.shape == std::vector<std::uint64_t>{64, 3},
        "Smedja did not regrow a released dynamic-mesh workspace.");
}

void test_execution_context_pool_bounds_contexts_and_bytes() {
    foamnordic::inference::ExecutionContextPool pool({2, 64});
    auto first = pool.acquire(20);
    auto second = pool.acquire(20);
    auto snapshot = pool.snapshot();
    require(
        snapshot.active_contexts == 2
            && snapshot.peak_active_contexts == 2
            && snapshot.admitted_bytes == 40,
        "Execution pool did not account for active contexts.");

    second = {};
    std::atomic<bool> admitted{false};
    std::exception_ptr failure;
    std::thread waiter([&] {
        try {
            auto third = pool.acquire(50);
            admitted.store(true, std::memory_order_release);
        } catch (...) {
            failure = std::current_exception();
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    require(
        !admitted.load(std::memory_order_acquire),
        "Execution pool exceeded its working-memory budget.");
    first = {};
    waiter.join();
    if (failure) {
        std::rethrow_exception(failure);
    }
    require(
        admitted.load(std::memory_order_acquire),
        "Execution pool did not admit work after resources were released.");
    require_throws(
        [&] { static_cast<void>(pool.acquire(65)); },
        "Execution pool accepted a request larger than its byte budget.");
    snapshot = pool.snapshot();
    require(
        snapshot.active_contexts == 0
            && snapshot.admitted_bytes == 0
            && snapshot.peak_admitted_bytes <= 64,
        "Execution pool did not release its admitted resources.");
}

void test_execution_context_pool_cancels_waiters() {
    foamnordic::inference::ExecutionContextPool pool({1, 64});
    auto active = pool.acquire(32);
    std::atomic<bool> cancelled{false};
    std::thread waiter([&] {
        try {
            static_cast<void>(pool.acquire(32));
        } catch (const std::runtime_error&) {
            cancelled.store(true, std::memory_order_release);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pool.cancel();
    waiter.join();
    require(
        cancelled.load(std::memory_order_acquire),
        "Execution pool did not cancel a blocked waiter.");
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

    foamnordic::inference::SmedjaWorkspace workspace;
    const auto& validated_contiguous = smedja.pack_validated_into(
        workspace, inputs, {1, 2, 3}, 22, 0.4);
    std::memcpy(
        contiguous_values.data(),
        validated_contiguous.bytes.data(),
        validated_contiguous.bytes.size());
    require_close(
        contiguous_values,
        {20.0, 30.0, 40.0},
        "Smedja changed a validated contiguous cell range.");

    const auto& validated_sparse = smedja.pack_validated_into(
        workspace, inputs, {0, 2, 4}, 23, 0.5);
    std::memcpy(
        sparse_values.data(),
        validated_sparse.bytes.data(),
        validated_sparse.bytes.size());
    require_close(
        sparse_values,
        {10.0, 30.0, 50.0},
        "Smedja changed a validated sparse cell selection.");
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
    explicit ConcurrentProbeKernel(bool concurrent = false)
        : concurrent_(concurrent) {}

    [[nodiscard]] foamnordic::inference::BackendCapabilities capabilities()
        const noexcept override {
        return {
            .concurrent_evaluation = concurrent_,
            .row_independent = true,
            .owns_compute_threads = false,
        };
    }

    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t /*rank*/) override {
        const auto active = active_.fetch_add(1) + 1;
        auto peak = peak_.load();
        while (active > peak && !peak_.compare_exchange_weak(peak, active)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        active_.fetch_sub(1);
        std::vector<double> values(features.shape.front(), 1.0);
        std::vector<std::byte> bytes(values.size() * sizeof(double));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        auto output = bindnordic::Tensor{
            "bindnordic.predictions",
            bindnordic::Element::float64,
            {features.shape.front(), 1},
            std::move(bytes)};
        static_cast<void>(exchange_index);
        static_cast<void>(physical_time);
        return output;
    }

    [[nodiscard]] std::uint32_t peak() const noexcept { return peak_.load(); }

private:
    bool concurrent_;
    std::atomic<std::uint32_t> active_{0};
    std::atomic<std::uint32_t> peak_{0};
};

class SharedCapacityProbeKernel final
    : public foamnordic::inference::PackedModelKernel {
public:
    SharedCapacityProbeKernel(
        foamnordic::inference::BackendCapabilities capabilities,
        std::atomic<std::uint32_t>& active,
        std::atomic<std::uint32_t>& peak)
        : capabilities_(capabilities), active_(active), peak_(peak) {}

    [[nodiscard]] foamnordic::inference::BackendCapabilities capabilities()
        const noexcept override {
        return capabilities_;
    }

    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t,
        double,
        std::uint32_t) override {
        const auto active = active_.fetch_add(1) + 1;
        auto peak = peak_.load();
        while (active > peak && !peak_.compare_exchange_weak(peak, active)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        active_.fetch_sub(1);
        std::vector<double> values(features.shape.front(), 1.0);
        std::vector<std::byte> bytes(values.size() * sizeof(double));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return {
            "bindnordic.predictions",
            bindnordic::Element::float64,
            {features.shape.front(), 1},
            std::move(bytes),
        };
    }

private:
    foamnordic::inference::BackendCapabilities capabilities_;
    std::atomic<std::uint32_t>& active_;
    std::atomic<std::uint32_t>& peak_;
};

foamnordic::inference::TensorMap capacity_inputs() {
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
    inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
    inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
    return inputs;
}

foamnordic::inference::ModelArtifact capacity_artifact() {
    return {
        foamnordic::inference::ModelFormat::onnx,
        "artifact/model.onnx",
        contract(),
        {},
        std::nullopt,
        std::nullopt,
    };
}

void test_artifact_kernel_rejects_unsupported_memory_boundaries() {
    std::atomic<std::uint32_t> active{0};
    std::atomic<std::uint32_t> peak{0};
    auto capabilities = foamnordic::inference::BackendCapabilities{};
    capabilities.input_memory = bindnordic::MemorySpace::device;
    SharedCapacityProbeKernel device_backend(capabilities, active, peak);
    require_throws(
        [&] {
            foamnordic::inference::ArtifactModelKernel kernel(
                capacity_artifact(), device_backend);
        },
        "ModelHost accepted a device input without a device tensor boundary.");

    capabilities = {};
    capabilities.completion = bindnordic::CompletionMode::event;
    SharedCapacityProbeKernel event_backend(capabilities, active, peak);
    require_throws(
        [&] {
            foamnordic::inference::ArtifactModelKernel kernel(
                capacity_artifact(), event_backend);
        },
        "ModelHost accepted event completion without an event boundary.");

    capabilities = {};
    capabilities.alignment_bytes = 3;
    SharedCapacityProbeKernel invalid_backend(capabilities, active, peak);
    require_throws(
        [&] {
            foamnordic::inference::ArtifactModelKernel kernel(
                capacity_artifact(), invalid_backend);
        },
        "ModelHost accepted an invalid backend alignment.");
}

void test_artifact_kernels_share_idle_execution_capacity() {
    auto capacity = std::make_shared<
        foamnordic::inference::ExecutionCapacityPool>(2);
    std::atomic<std::uint32_t> active{0};
    std::atomic<std::uint32_t> peak{0};
    const auto capabilities = foamnordic::inference::BackendCapabilities{
        .concurrent_evaluation = true,
        .row_independent = true,
        .owns_compute_threads = false,
    };
    SharedCapacityProbeKernel first_backend(capabilities, active, peak);
    SharedCapacityProbeKernel second_backend(capabilities, active, peak);
    auto resources = foamnordic::inference::ExecutionResources{2, 4096};
    resources.capacity = capacity;
    resources.capacity_units = 2;
    foamnordic::inference::ArtifactModelKernel first(
        capacity_artifact(), first_backend, resources);
    foamnordic::inference::ArtifactModelKernel second(
        capacity_artifact(), second_backend, resources);
    auto first_request = std::async(std::launch::async, [&] {
        return first.evaluate(capacity_inputs(), {0, 1}, 1, 0.1);
    });
    auto second_request = std::async(std::launch::async, [&] {
        return second.evaluate(capacity_inputs(), {0, 1}, 2, 0.1);
    });
    first_request.get();
    second_request.get();
    require(
        peak.load() == 2 && capacity->snapshot().peak_active == 2,
        "Independent kernels did not borrow idle shared execution capacity.");
}

void test_thread_owning_backends_reserve_their_full_capacity() {
    auto capacity = std::make_shared<
        foamnordic::inference::ExecutionCapacityPool>(2);
    std::atomic<std::uint32_t> active{0};
    std::atomic<std::uint32_t> peak{0};
    const auto capabilities = foamnordic::inference::BackendCapabilities{
        .concurrent_evaluation = true,
        .row_independent = true,
        .owns_compute_threads = true,
    };
    SharedCapacityProbeKernel first_backend(capabilities, active, peak);
    SharedCapacityProbeKernel second_backend(capabilities, active, peak);
    auto resources = foamnordic::inference::ExecutionResources{2, 4096};
    resources.capacity = capacity;
    resources.capacity_units = 2;
    foamnordic::inference::ArtifactModelKernel first(
        capacity_artifact(), first_backend, resources);
    foamnordic::inference::ArtifactModelKernel second(
        capacity_artifact(), second_backend, resources);
    auto first_request = std::async(std::launch::async, [&] {
        return first.evaluate(capacity_inputs(), {0, 1}, 1, 0.1);
    });
    auto second_request = std::async(std::launch::async, [&] {
        return second.evaluate(capacity_inputs(), {0, 1}, 2, 0.1);
    });
    first_request.get();
    second_request.get();
    require(
        peak.load() == 1 && capacity->snapshot().peak_active == 2,
        "Thread-owning backends oversubscribed shared execution capacity.");
}

void test_artifact_kernel_owns_a_narrow_backend_lock() {
    ConcurrentProbeKernel packed_kernel;
    foamnordic::inference::ArtifactModelKernel kernel(
        {
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
    require(
        kernel.execution_snapshot().peak_active_contexts == 1,
        "Artifact kernel admitted concurrent work for a serialized backend.");
}

void test_reentrant_backend_uses_bounded_execution_contexts() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::compiled,
            "artifact/model.cpp",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        packed_kernel,
        {2, 4096});
    const auto invoke = [&](std::uint64_t exchange_index) {
        foamnordic::inference::TensorMap inputs;
        inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
        inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
        inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
        static_cast<void>(kernel.evaluate(inputs, {0, 1}, exchange_index, 0.25));
    };
    std::thread first([&] { invoke(20); });
    std::thread second([&] { invoke(21); });
    first.join();
    second.join();
    require(
        packed_kernel.peak() == 2,
        "A reentrant backend did not use its execution contexts concurrently.");
}

void test_artifact_kernel_rejects_an_oversized_working_set() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::onnx,
            "artifact/model.onnx",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
        },
        packed_kernel,
        {1, 79});
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
    inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
    inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
    require_throws(
        [&] {
            static_cast<void>(kernel.evaluate(inputs, {0, 1}, 30, 0.25));
        },
        "Artifact kernel accepted a request above its working-memory limit.");
    require(
        packed_kernel.peak() == 0,
        "An oversized request reached the model backend before rejection.");
}

void test_artifact_kernel_samples_stage_costs_on_request() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::compiled,
            "artifact/model.cpp",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        packed_kernel,
        {1, 4096, 2});
    for (std::uint64_t exchange_index = 40; exchange_index < 43; ++exchange_index) {
        foamnordic::inference::TensorMap inputs;
        inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
        inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
        inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
        static_cast<void>(
            kernel.evaluate(inputs, {0, 1}, exchange_index, 0.25));
    }
    require(
        kernel.execution_snapshot().profile_samples == 2,
        "Artifact kernel did not honour its profiling sample interval.");
}

void test_artifact_kernel_profiles_matching_exchanges_for_every_rank() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::compiled,
            "artifact/model.cpp",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        packed_kernel,
        {1, 4096, 2});
    std::ostringstream logs;
    auto* previous = std::clog.rdbuf(logs.rdbuf());
    for (std::uint64_t exchange_index = 40; exchange_index < 42; ++exchange_index) {
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            foamnordic::inference::TensorMap inputs;
            inputs.emplace("c_tilde", scalar_tensor("c_tilde", {0.1, 0.2}));
            inputs.emplace("c_var", scalar_tensor("c_var", {0.01, 0.02}));
            inputs.emplace("T_tilde", scalar_tensor("T_tilde", {300.0, 400.0}));
            static_cast<void>(
                kernel.evaluate(inputs, {0, 1}, exchange_index, 0.25, rank));
        }
    }
    std::clog.rdbuf(previous);

    const auto output = logs.str();
    require(
        kernel.execution_snapshot().profile_samples == 2,
        "Artifact kernel did not sample every rank at the selected exchange.");
    require(
        output.find("\"rank\":0,\"exchange_index\":40") != std::string::npos
            && output.find("\"rank\":1,\"exchange_index\":40")
                   != std::string::npos,
        "Model-work telemetry did not identify both sampled ranks.");
    require(
        output.find("\"rows\":2,\"active_cells\":2,\"arrived_ns\":")
                != std::string::npos
            && output.find("\"admitted_ns\":") != std::string::npos
            && output.find("\"execution_started_ns\":") != std::string::npos
            && output.find("\"completed_ns\":") != std::string::npos,
        "Model-work telemetry omitted scheduling timestamps or workload size.");
    require(
        output.find("\"exchange_index\":41") == std::string::npos,
        "Model-work telemetry sampled an exchange outside the interval.");
}

void test_artifact_kernel_records_completed_workload() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::compiled,
            "artifact/model.cpp",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        packed_kernel,
        {1, 4096, 1});
    kernel.observe_exchange(100, 80, 2400, 800);
    kernel.observe_exchange(120, 90, 2880, 960);
    const auto snapshot = kernel.execution_snapshot();
    require(
        snapshot.completed_exchanges == 2
            && snapshot.rows == 220
            && snapshot.active_cells == 170
            && snapshot.input_payload_bytes == 5280
            && snapshot.output_payload_bytes == 1760,
        "Artifact kernel did not accumulate completed workload telemetry.");
    require(
        snapshot.peak_rows == 120
            && snapshot.peak_active_cells == 90
            && snapshot.peak_input_payload_bytes == 2880
            && snapshot.peak_output_payload_bytes == 960,
        "Artifact kernel did not retain per-exchange workload peaks.");
}

void test_artifact_kernel_disables_workload_telemetry_by_default() {
    ConcurrentProbeKernel packed_kernel(true);
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::compiled,
            "artifact/model.cpp",
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        packed_kernel,
        {1, 4096, 0});
    require(
        !kernel.telemetry_enabled(),
        "Artifact kernel enabled telemetry for profile_every=0.");
    kernel.observe_exchange(100, 80, 2400, 800);
    const auto snapshot = kernel.execution_snapshot();
    require(
        snapshot.completed_exchanges == 0
            && snapshot.rows == 0
            && snapshot.active_cells == 0
            && snapshot.input_payload_bytes == 0
            && snapshot.output_payload_bytes == 0,
        "Disabled telemetry updated workload counters.");
}

class LoadedProbeKernel final : public foamnordic::inference::PackedModelKernel {
public:
    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t /*rank*/) override {
        std::vector<double> values(features.shape.front(), 1.0);
        std::vector<std::byte> bytes(values.size() * sizeof(double));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        auto output = bindnordic::Tensor{
            "bindnordic.predictions",
            bindnordic::Element::float64,
            {features.shape.front(), 1},
            std::move(bytes)};
        static_cast<void>(exchange_index);
        static_cast<void>(physical_time);
        return output;
    }
};

class LoadedProbeConnector final : public foamnordic::backend::ModelConnector {
public:
    [[nodiscard]] std::string_view id() const noexcept override {
        return "test-loaded-compiled";
    }

    [[nodiscard]] bool supports(
        foamnordic::inference::ModelFormat format) const noexcept override {
        return format == foamnordic::inference::ModelFormat::compiled;
    }

    [[nodiscard]] std::unique_ptr<foamnordic::inference::PackedModelKernel> load(
        std::span<const std::byte> /*payload*/,
        const foamnordic::inference::ModelArtifact& /*artifact*/,
        std::uint32_t /*threads*/) const override {
        return std::make_unique<LoadedProbeKernel>();
    }
};

void test_loaded_artifact_preserves_backend_synchronization_ownership() {
    foamnordic::backend::register_connector(
        std::make_unique<LoadedProbeConnector>());
    const auto path = std::filesystem::temp_directory_path()
                      / ("foamnordic-loaded-kernel-"
                         + std::to_string(std::random_device{}()) + ".fnom");
    const auto payload = path.string() + ".payload";
    { std::ofstream stream(payload, std::ios::binary); stream << "compiled"; }
    foamnordic::inference::write_bundle(
        path,
        {
            foamnordic::inference::ModelFormat::compiled,
            std::filesystem::path(payload).filename().string(),
            contract(),
            {},
            std::nullopt,
            std::nullopt,
            "compiled",
        },
        payload);
    auto loaded = foamnordic::inference::load_model(path);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(payload, ignored);
    require(
        loaded.kernel->owns_backend_synchronization(),
        "The real artifact loader hid the kernel synchronization boundary.");
}

void test_table_bundle_loads_through_the_model_boundary() {
    std::vector<std::byte> table;
    for (const char character : std::string("FNOTAB1\0", 8)) {
        append_value(table, character);
    }
    append_value(table, std::uint8_t{2});
    append_value(table, std::uint8_t{2});
    append_value(table, std::uint16_t{0});
    append_value(table, std::uint64_t{2});
    for (const auto& axis : std::array<std::vector<double>, 2>{
             std::vector<double>{0.0, 1.0},
             std::vector<double>{0.0, 2.0}}) {
        append_value(table, std::uint8_t{0});
        append_value(table, std::uint64_t{axis.size()});
        for (const auto point : axis) append_value(table, point);
    }
    for (const auto point : std::vector<double>{
             0.0, 10.0,
             4.0, 14.0,
             2.0, 12.0,
             6.0, 16.0}) {
        append_value(table, point);
    }

    const auto root = std::filesystem::temp_directory_path();
    const auto nonce = std::to_string(std::random_device{}());
    const auto payload = root / ("foamnordic-table-" + nonce + ".table");
    const auto bundle = root / ("foamnordic-table-" + nonce + ".fnom");
    write_bytes(payload, table);
    foamnordic::inference::write_bundle(
        bundle,
        {
            foamnordic::inference::ModelFormat::table,
            payload.filename().string(),
            {
                "table",
                {
                    {"x", foamnordic::fjord::Element::float64, 1},
                    {"z", foamnordic::fjord::Element::float64, 1},
                },
                {{"state", foamnordic::fjord::Element::float64, 2}},
            },
            {},
            std::nullopt,
            std::nullopt,
            "table",
        },
        payload);

    auto loaded = foamnordic::inference::load_model(bundle);
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("x", scalar_tensor("x", {0.25, 2.0}));
    inputs.emplace("z", scalar_tensor("z", {0.5, -1.0}));
    const auto outputs = loaded.kernel->evaluate(inputs, {0, 1}, 0, 0.0);
    const auto& output = outputs.at("state");
    std::vector<double> actual(output.bytes.size() / sizeof(double));
    std::memcpy(actual.data(), output.bytes.data(), output.bytes.size());
    require_close(actual, {1.5, 11.5, 2.0, 12.0}, "Loaded table interpolation failed.");

    std::error_code ignored;
    std::filesystem::remove(payload, ignored);
    std::filesystem::remove(bundle, ignored);
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
    const auto standalone_path = std::filesystem::temp_directory_path()
                                 / ("foamnordic-standalone-"
                                    + std::to_string(std::random_device{}()) + ".fnom");
    {
        std::ofstream stream(standalone_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(encoded.data()),
            static_cast<std::streamsize>(encoded.size()));
    }
    require_throws(
        [&] { static_cast<void>(foamnordic::inference::read_manifest(standalone_path)); },
        "FNOM accepted metadata without its self-contained bundle.");
    std::error_code ignored;
    std::filesystem::remove(standalone_path, ignored);
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
        accelerated_round_trip.runtime == "sklearnex",
        "Manifest did not round-trip the Joblib execution runtime.");

    foamnordic::inference::ModelArtifact compiled{
        foamnordic::inference::ModelFormat::compiled,
        "models/closure.cpp",
        contract(),
        {},
        std::nullopt,
        std::nullopt,
        "compiled",
    };
    const auto compiled_round_trip = foamnordic::inference::decode_manifest(
        foamnordic::inference::encode_manifest(compiled));
    require(
        compiled_round_trip.format
                == foamnordic::inference::ModelFormat::compiled
            && compiled_round_trip.runtime == "compiled",
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

    std::error_code ignored;
    std::filesystem::remove(payload_path, ignored);
    std::filesystem::remove(bundle_path, ignored);
    std::filesystem::remove(malformed_path, ignored);
    std::filesystem::remove(extracted_path, ignored);
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
    test_execution_context_pool_bounds_contexts_and_bytes();
    test_execution_context_pool_cancels_waiters();
    test_smedja_packs_contiguous_and_sparse_cell_ranges();
    test_smedja_fuses_float_input_conversion_with_packing();
    test_artifact_kernel_owns_a_narrow_backend_lock();
    test_reentrant_backend_uses_bounded_execution_contexts();
    test_artifact_kernels_share_idle_execution_capacity();
    test_thread_owning_backends_reserve_their_full_capacity();
    test_artifact_kernel_rejects_unsupported_memory_boundaries();
    test_artifact_kernel_rejects_an_oversized_working_set();
    test_artifact_kernel_samples_stage_costs_on_request();
    test_artifact_kernel_profiles_matching_exchanges_for_every_rank();
    test_artifact_kernel_records_completed_workload();
    test_artifact_kernel_disables_workload_telemetry_by_default();
    test_loaded_artifact_preserves_backend_synchronization_ownership();
    test_table_bundle_loads_through_the_model_boundary();
    test_dense_affine_kernel_float64();
    test_dense_affine_kernel_float32();
    test_manifest_round_trip_and_corruption_rejection();
    test_self_contained_bundle_round_trip();
}
