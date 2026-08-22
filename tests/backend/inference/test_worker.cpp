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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

foamnordic::fjord::Tensor scalar_field(
    std::string name,
    const std::vector<double>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        foamnordic::fjord::Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        4,
        0.2,
    };
}

std::vector<double> values_of(const foamnordic::fjord::Tensor& tensor) {
    std::vector<double> values(tensor.bytes.size() / sizeof(double));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

void test_resident_worker_lifecycle() {
    const auto socket_path = std::filesystem::temp_directory_path()
                             / ("foamnordic-worker-"
                                + std::to_string(std::random_device{}()) + ".sock");
    const foamnordic::closure::ClosureContract contract{
        "resident-affine",
        {
            {"a", foamnordic::fjord::Element::float64, 1},
            {"b", foamnordic::fjord::Element::float64, 1},
        },
        {{"result", foamnordic::fjord::Element::float64, 1}},
    };
    foamnordic::closure::DenseAffineKernel affine(2, 1, {2.0, -1.0}, {0.5});
    foamnordic::closure::ArtifactModelKernel kernel(
        {
            1,
            foamnordic::closure::ModelFormat::onnx,
            "reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        },
        affine);
    foamnordic::closure::EvaluateEveryCell bypass;
    std::exception_ptr worker_failure;

    {
        foamnordic::closure::NativeClosureWorker worker(
            foamnordic::fjord::FjordAddress::local(socket_path.string()),
            {
                1,
                foamnordic::closure::ModelFormat::onnx,
                "reference.onnx",
                contract,
                {},
                std::nullopt,
                std::nullopt,
            },
            bypass,
            kernel);
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
        client.send(scalar_field("a", {1.0, 3.0}).view());
        client.send(scalar_field("b", {4.0, 5.0}).view());
        client.complete(4, 2);

        auto message = client.receive_message();
        require(
            message.kind == foamnordic::fjord::RuneKind::tensor
                && message.tensor.has_value(),
            "Resident worker did not return a prediction tensor.");
        const auto values = values_of(*message.tensor);
        require(
            values.size() == 2
                && std::abs(values[0] + 1.5) < 1.0e-12
                && std::abs(values[1] - 1.5) < 1.0e-12,
            "Resident worker returned incorrect affine predictions.");
        std::uint64_t completed_exchange = 0;
        require(
            client.receive_control(&completed_exchange)
                    == foamnordic::fjord::RuneKind::complete
                && completed_exchange == 4,
            "Resident worker returned an incorrect atomic completion.");
        client.shutdown();
        worker_thread.join();
        if (worker_failure) {
            std::rethrow_exception(worker_failure);
        }
    }
    require(
        !std::filesystem::exists(socket_path),
        "Resident worker left its Unix socket behind.");
}

}  // namespace

int main() {
    test_resident_worker_lifecycle();
}
