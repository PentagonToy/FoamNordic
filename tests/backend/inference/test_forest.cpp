#include "foamnordic/backend/inference/forest.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/harbor.hpp"
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <unistd.h>

using namespace foamnordic;
static void word(std::vector<std::byte>& data, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) data.push_back(std::byte((value >> (i * 8)) & 255));
}
int main() {
    std::vector<std::byte> data(8);
    std::memcpy(data.data(), "FNOFOR1\0", 8);
    word(data, 1); word(data, 1); word(data, 3); word(data, 0);
    for (int node = 0; node < 3; ++node) {
        word(data, node ? -1 : 1); word(data, node ? -1 : 2);
        word(data, 0); word(data, std::bit_cast<std::uint64_t>(0.5));
        word(data, std::bit_cast<std::uint64_t>(double(node))); word(data, 1);
    }
    inference::ForestKernel kernel(data, 3);
    std::vector<double> values(1000, 0.0);
    values[1] = 1.0; values[2] = std::numeric_limits<double>::quiet_NaN();
    fjord::Tensor input{"x", fjord::Element::float64, {1000, 1},
        std::vector<std::byte>(values.size() * sizeof(double)), 0, 0};
    std::memcpy(input.bytes.data(), values.data(), input.bytes.size());
    for (int call = 0; call < 100; ++call) {
        auto result = kernel.evaluate(input.view(), call, 0);
        for (unsigned row = 0; row < values.size(); ++row) {
            double value; std::memcpy(&value, result.bytes.data() + row * 8, 8);
            if (value != (row == 1 ? 2.0 : 1.0)) throw std::runtime_error("Forest routing mismatch.");
        }
    }
    auto reject = [](const auto& bytes) {
        try { inference::ForestKernel invalid(bytes); }
        catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("Malformed forest accepted.");
    };
    auto invalid = data; invalid.pop_back(); reject(invalid);
    invalid = data; invalid[0] = std::byte{0}; reject(invalid);
    invalid = data; invalid[40] = std::byte{0}; reject(invalid); // Self-cycle.
    invalid = data; invalid[32] = std::byte{3}; reject(invalid); // Root outside nodes.
    invalid = data; invalid[80] = std::byte{2}; reject(invalid); // Invalid missing flag.

    const auto root = std::filesystem::temp_directory_path()
        / ("fno-forest-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(root);
    const auto payload = root / "model.data";
    { std::ofstream stream(payload, std::ios::binary);
      stream.write(reinterpret_cast<const char*>(data.data()), data.size()); }
    inference::ModelArtifact artifact;
    artifact.format = inference::ModelFormat::compiled;
    artifact.runtime = "forest";
    artifact.artifact_path = "model.data";
    artifact.contract = {"forest", {{"x", fjord::Element::float64, 1}},
                         {{"y", fjord::Element::float64, 1}}};
    const auto bundle = root / "model.fnom";
    inference::write_bundle(bundle, artifact, payload);
    std::filesystem::remove(payload); // The loader must use only the bundle.
    input.shape = {1000};
    for (bool shm : {false, true}) {
        inference::EvaluateAllCells bypass;
        inference::WorkerOptions options;
        options.shared_memory = shm;
        options.model_threads = 3;
        inference::ModelWorker worker(fjord::FjordAddress::local((root / "socket").string()),
                                     bundle, bypass, options);
        std::exception_ptr failure;
        std::thread thread([&] { try { worker.run(); } catch (...) { failure = std::current_exception(); } });
        fjord::Harbor client(fjord::connect(worker.address()));
        auto session = client.connect_session({65536,
            shm ? fjord::Capability::uds | fjord::Capability::shm : fjord::Capability::uds,
            0, 1, 65536});
        if (fjord::any(session.capabilities & fjord::Capability::shm)) client.accept_shared_memory();
        for (unsigned call = 0; call < 4; ++call) {
            input.time_index = call;
            client.send(input.view());
            client.complete(call, 1);
            auto result = client.receive_message();
            if (!result.tensor || result.tensor->bytes.size() != values.size() * 8)
                throw std::runtime_error("Forest worker response missing.");
            double prediction;
            std::memcpy(&prediction, result.tensor->bytes.data() + 8, 8);
            if (prediction != 2.0 || client.receive_control() != fjord::RuneKind::complete)
                throw std::runtime_error("Forest worker response mismatch.");
        }
        client.shutdown(); thread.join();
        if (failure) std::rethrow_exception(failure);
    }
    std::filesystem::remove_all(root);
}
