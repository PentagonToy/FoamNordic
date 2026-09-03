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

#include "foamnordic/backend/inference/onnx.hpp"
#include "foamnordic/backend/inference/load.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace {

constexpr std::array<unsigned char, 192> model_bytes{
    0x08, 0x0b, 0x3a, 0xb5, 0x01, 0x0a, 0x1d, 0x0a, 0x08, 0x66, 0x65, 0x61,
    0x74, 0x75, 0x72, 0x65, 0x73, 0x0a, 0x01, 0x57, 0x12, 0x06, 0x6c, 0x69,
    0x6e, 0x65, 0x61, 0x72, 0x22, 0x06, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c,
    0x0a, 0x1d, 0x0a, 0x06, 0x6c, 0x69, 0x6e, 0x65, 0x61, 0x72, 0x0a, 0x01,
    0x62, 0x12, 0x0b, 0x70, 0x72, 0x65, 0x64, 0x69, 0x63, 0x74, 0x69, 0x6f,
    0x6e, 0x73, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12, 0x0e, 0x46, 0x6f, 0x61,
    0x6d, 0x4e, 0x6f, 0x72, 0x64, 0x69, 0x63, 0x54, 0x65, 0x73, 0x74, 0x2a,
    0x13, 0x08, 0x02, 0x08, 0x01, 0x10, 0x01, 0x22, 0x08, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x00, 0x80, 0xbf, 0x42, 0x01, 0x57, 0x2a, 0x0d, 0x08, 0x01,
    0x10, 0x01, 0x22, 0x04, 0x00, 0x00, 0x00, 0x3f, 0x42, 0x01, 0x62, 0x5a,
    0x1e, 0x0a, 0x08, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x12,
    0x12, 0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c, 0x0a, 0x06, 0x12, 0x04, 0x63,
    0x65, 0x6c, 0x6c, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x21, 0x0a, 0x0b, 0x70,
    0x72, 0x65, 0x64, 0x69, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x12, 0x12,
    0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c, 0x0a, 0x06, 0x12, 0x04, 0x63, 0x65,
    0x6c, 0x6c, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
};

std::vector<std::byte> bytes(std::span<const float> values) {
    std::vector<std::byte> result(values.size_bytes());
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

}  // namespace

int main() {
    const auto model_path =
        std::filesystem::temp_directory_path() / "foamnordic-native-test.onnx";
    {
        std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
        output.write(
            reinterpret_cast<const char*>(model_bytes.data()),
            static_cast<std::streamsize>(model_bytes.size()));
    }

    foamnordic::inference::OnnxPackedKernel kernel(model_path);
    constexpr std::array<float, 4> values{1.0F, 3.0F, -2.0F, 4.0F};
    foamnordic::fjord::Tensor features{
        "features",
        foamnordic::fjord::Element::float32,
        {2, 2},
        bytes(values),
        7,
        0.25,
    };
    const auto original_feature_bytes = features.bytes;

    const auto output = kernel.evaluate(features.view(), 7, 0.25);
    assert(features.bytes == original_feature_bytes);
    assert((output.shape == std::vector<std::uint64_t>{2, 1}));
    std::array<float, 2> predictions{};
    std::memcpy(predictions.data(), output.bytes.data(), output.bytes.size());
    assert(std::abs(predictions[0] - (-0.5F)) < 1.0e-6F);
    assert(std::abs(predictions[1] - (-7.5F)) < 1.0e-6F);

    std::vector<std::byte> unaligned_storage(features.bytes.size() + 1);
    std::memcpy(
        unaligned_storage.data() + 1,
        features.bytes.data(),
        features.bytes.size());
    const foamnordic::fjord::TensorView unaligned_features{
        features.name,
        features.element,
        features.shape,
        std::span<const std::byte>(unaligned_storage).subspan(1),
        features.time_index,
        features.physical_time,
        features.solver_time_index,
    };
    const auto unaligned_output =
        kernel.evaluate(unaligned_features, 7, 0.25);
    assert(unaligned_output.bytes == output.bytes);

    const auto manifest_path =
        std::filesystem::temp_directory_path() / "foamnordic-native-test.fnom";
    foamnordic::inference::write_bundle(
        manifest_path,
        {
            1,
            foamnordic::inference::ModelFormat::onnx,
            model_path.filename().string(),
            {
                "native-onnx",
                {{"features", foamnordic::fjord::Element::float32, 2}},
                {{"predictions", foamnordic::fjord::Element::float32, 1}},
            },
            {},
            foamnordic::inference::AffineScaler::standard(
                {1.0, 1.0}, {2.0, 2.0}),
            foamnordic::inference::AffineScaler::robust({10.0}, {2.0}),
        },
        model_path);
    std::filesystem::remove(model_path);
    auto loaded = foamnordic::inference::load_model(manifest_path);
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("features", features);
    const auto loaded_outputs =
        loaded.kernel->evaluate(inputs, {0, 1}, 7, 0.25);
    const auto& loaded_output = loaded_outputs.at("predictions");
    std::array<float, 2> loaded_predictions{};
    std::memcpy(
        loaded_predictions.data(),
        loaded_output.bytes.data(),
        loaded_output.bytes.size());
    assert(std::abs(loaded_predictions[0] - 9.0F) < 1.0e-6F);
    assert(std::abs(loaded_predictions[1] - 2.0F) < 1.0e-6F);

    const auto socket_path =
        std::filesystem::temp_directory_path()
        / ("foamnordic-onnx-worker-" + std::to_string(std::random_device{}())
           + ".sock");
    foamnordic::inference::EvaluateAllCells bypass;
    foamnordic::inference::ModelWorker worker(
        foamnordic::fjord::FjordAddress::local(socket_path.string()),
        manifest_path,
        bypass);
    std::exception_ptr worker_failure;
    std::thread worker_thread([&] {
        try {
            worker.run();
        } catch (...) {
            worker_failure = std::current_exception();
        }
    });
    foamnordic::fjord::Harbor client(
        foamnordic::fjord::connect(worker.address()));
    const auto session = client.connect_session({
        2048,
        foamnordic::fjord::Capability::uds
            | foamnordic::fjord::Capability::shm,
        0,
        1,
        4096,
    });
    if (foamnordic::fjord::any(
            session.capabilities & foamnordic::fjord::Capability::shm)) {
        client.accept_shared_memory();
    }
    client.send(features.view());
    client.complete(7, 1);
    const auto prediction_message = client.receive_message();
    assert(prediction_message.tensor.has_value());
    std::array<float, 2> worker_predictions{};
    std::memcpy(
        worker_predictions.data(),
        prediction_message.tensor->bytes.data(),
        prediction_message.tensor->bytes.size());
    assert(std::abs(worker_predictions[0] - 9.0F) < 1.0e-6F);
    assert(std::abs(worker_predictions[1] - 2.0F) < 1.0e-6F);
    std::uint64_t completed_exchange = 0;
    assert(
        client.receive_control(&completed_exchange)
        == foamnordic::fjord::RuneKind::complete);
    assert(completed_exchange == 7);
    client.shutdown();
    worker_thread.join();
    if (worker_failure) {
        std::rethrow_exception(worker_failure);
    }
    assert(!std::filesystem::exists(socket_path));

    std::filesystem::remove(manifest_path);
}
