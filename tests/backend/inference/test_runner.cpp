#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "foamnordic/backend/inference/runner.hpp"
#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/fjord/channel.hpp"

namespace {

using foamnordic::closure::TensorMap;
using foamnordic::fjord::Element;
using foamnordic::fjord::Tensor;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Tensor scalar_field(
    std::string name,
    const std::vector<double>& values,
    std::uint64_t exchange_index,
    double physical_time) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        exchange_index,
        physical_time,
    };
}

std::vector<double> values_of(const Tensor& tensor) {
    std::vector<double> values(tensor.bytes.size() / sizeof(double));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

class CombustionBypass final : public foamnordic::closure::BypassPolicy {
public:
    std::vector<std::uint64_t> prepare(
        const TensorMap& inputs,
        TensorMap& outputs,
        std::uint64_t cell_count) const override {
        const auto variance = values_of(inputs.at("c_var"));
        std::vector<std::uint64_t> active;
        for (std::uint64_t cell = 0; cell < cell_count; ++cell) {
            if (variance[cell] > 0.05) {
                active.push_back(cell);
            }
        }
        const auto& source = inputs.at("c_var");
        outputs.emplace(
            "omega_c",
            scalar_field(
                "omega_c",
                std::vector<double>(cell_count, 0.0),
                source.time_index,
                source.physical_time));
        return active;
    }

    void merge(
        TensorMap& predictions,
        TensorMap& outputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t) const override {
        const auto& prediction = predictions.at("omega_c");
        auto predicted = values_of(prediction);
        auto merged = values_of(outputs.at("omega_c"));
        for (std::size_t index = 0; index < active_cells.size(); ++index) {
            merged[active_cells[index]] = predicted[index];
        }
        outputs.insert_or_assign(
            "omega_c",
            scalar_field(
                "omega_c",
                merged,
                prediction.time_index,
                prediction.physical_time));
    }
};

class CombustionKernel final : public foamnordic::closure::ModelKernel {
public:
    TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time) override {
        require(inputs.size() == 3, "Kernel did not receive the complete input batch.");
        require(
            active_cells == std::vector<std::uint64_t>{1, 3},
            "Kernel received an incorrect active-cell mask.");
        TensorMap outputs;
        outputs.emplace(
            "omega_c",
            scalar_field("omega_c", {10.0, 20.0}, exchange_index, physical_time));
        return outputs;
    }
};

void test_native_combustion_exchange(bool shared_memory) {
    auto channels = shared_memory
                        ? foamnordic::fjord::shared_memory_channel_pair(8, 256)
                        : foamnordic::fjord::local_channel_pair();
    auto client_channel = std::move(channels.first);
    auto worker_channel = std::move(channels.second);
    foamnordic::fjord::Harbor client(std::move(client_channel));
    foamnordic::fjord::Harbor worker(std::move(worker_channel));
    CombustionBypass bypass;
    CombustionKernel kernel;
    std::exception_ptr worker_failure;

    std::thread worker_thread([&] {
        try {
            static_cast<void>(worker.accept_session({
                1,
                foamnordic::fjord::Capability::uds,
                0,
                1,
                4096,
            }));
            foamnordic::closure::NativeClosureRunner runner(
                worker,
                {
                    "progress-variable-combustion",
                    {
                        {"c_tilde", Element::float64, 1},
                        {"c_var", Element::float64, 1},
                        {"T_tilde", Element::float64, 1},
                    },
                    {{"omega_c", Element::float64, 1}},
                },
                bypass,
                kernel);
            require(runner.run_one(), "Runner did not process the combustion exchange.");
            require(!runner.run_one(), "Runner did not stop at the shutdown boundary.");
        } catch (...) {
            worker_failure = std::current_exception();
        }
    });

    static_cast<void>(client.connect_session({
        7001,
        foamnordic::fjord::Capability::uds,
        0,
        1,
        4096,
    }));
    for (const auto& tensor : {
             scalar_field("c_tilde", {0.1, 0.4, 0.7, 0.9}, 8, 0.25),
             scalar_field("c_var", {0.0, 0.1, 0.01, 0.2}, 8, 0.25),
             scalar_field("T_tilde", {300.0, 900.0, 1200.0, 1600.0}, 8, 0.25),
         }) {
        client.send(tensor.view());
    }
    client.complete(8, 3);

    auto output_message = client.receive_message();
    require(
        output_message.kind == foamnordic::fjord::RuneKind::tensor
            && output_message.tensor.has_value(),
        "Client did not receive a closure output tensor.");
    require(
        values_of(*output_message.tensor) == std::vector<double>{0.0, 10.0, 0.0, 20.0},
        "Native closure runner returned an incorrect bypass merge.");
    std::uint64_t completed_exchange = 0;
    require(
        client.receive_control(&completed_exchange) == foamnordic::fjord::RuneKind::complete
            && completed_exchange == 8,
        "Native closure runner returned an incorrect completion boundary.");
    client.shutdown();
    worker_thread.join();
    if (worker_failure) {
        std::rethrow_exception(worker_failure);
    }
}

void test_scaled_affine_exchange(bool shared_memory) {
    auto channels = shared_memory
                        ? foamnordic::fjord::shared_memory_channel_pair(8, 256)
                        : foamnordic::fjord::local_channel_pair();
    foamnordic::fjord::Harbor client(std::move(channels.first));
    foamnordic::fjord::Harbor worker(std::move(channels.second));
    foamnordic::closure::EvaluateEveryCell bypass;
    foamnordic::closure::DenseAffineKernel affine(
        2,
        1,
        {2.0, 3.0},
        {1.0});
    foamnordic::closure::ArtifactModelKernel kernel(
        {
            1,
            foamnordic::closure::ModelFormat::onnx,
            "artifact/reference-affine.onnx",
            {
                "scaled-affine-reference",
                {
                    {"feature_a", Element::float64, 1},
                    {"feature_b", Element::float64, 1},
                },
                {{"target", Element::float64, 1}},
            },
            {},
            foamnordic::closure::AffineScaler::standard(
                {10.0, 100.0},
                {2.0, 10.0}),
            foamnordic::closure::AffineScaler::standard(
                {5.0},
                {2.0}),
        },
        affine);
    std::exception_ptr worker_failure;

    std::thread worker_thread([&] {
        try {
            static_cast<void>(worker.accept_session({
                1,
                foamnordic::fjord::Capability::uds,
                0,
                1,
                4096,
            }));
            foamnordic::closure::NativeClosureRunner runner(
                worker,
                {
                    "scaled-affine-reference",
                    {
                        {"feature_a", Element::float64, 1},
                        {"feature_b", Element::float64, 1},
                    },
                    {{"target", Element::float64, 1}},
                },
                bypass,
                kernel);
            require(runner.run_one(), "Scaled affine runner did not process its exchange.");
            require(!runner.run_one(), "Scaled affine runner did not stop at shutdown.");
        } catch (...) {
            worker_failure = std::current_exception();
        }
    });

    static_cast<void>(client.connect_session({
        8001,
        foamnordic::fjord::Capability::uds,
        0,
        1,
        4096,
    }));
    client.send(scalar_field("feature_a", {10.0, 14.0}, 12, 0.5).view());
    client.send(scalar_field("feature_b", {100.0, 120.0}, 12, 0.5).view());
    client.complete(12, 2);

    auto output_message = client.receive_message();
    require(
        output_message.kind == foamnordic::fjord::RuneKind::tensor
            && output_message.tensor.has_value(),
        "Scaled affine runner did not return an output tensor.");
    const auto actual = values_of(*output_message.tensor);
    require(
        actual.size() == 2,
        "Scaled affine runner returned an incorrect prediction size.");
    require(
        std::abs(actual[0] - 7.0) < 1.0e-12
            && std::abs(actual[1] - 27.0) < 1.0e-12,
        "Scaled affine runner returned incorrect physical-space values: "
            + std::to_string(actual[0]) + ", " + std::to_string(actual[1]));
    std::uint64_t completed_exchange = 0;
    require(
        client.receive_control(&completed_exchange)
                == foamnordic::fjord::RuneKind::complete
            && completed_exchange == 12,
        "Scaled affine runner returned an incorrect completion boundary.");
    client.shutdown();
    worker_thread.join();
    if (worker_failure) {
        std::rethrow_exception(worker_failure);
    }
}

void test_worker_failure_is_reported_to_solver(bool sharedMemory) {
    auto channels = sharedMemory
                        ? foamnordic::fjord::shared_memory_channel_pair(8, 256)
                        : foamnordic::fjord::local_channel_pair();
    foamnordic::fjord::Harbor client(std::move(channels.first));
    foamnordic::fjord::Harbor worker(std::move(channels.second));
    foamnordic::closure::EvaluateEveryCell bypass;
    CombustionKernel kernel;
    std::exception_ptr workerFailure;

    std::thread workerThread([&] {
        try {
            static_cast<void>(worker.accept_session({
                1,
                foamnordic::fjord::Capability::uds,
                0,
                1,
                4096,
            }));
            foamnordic::closure::NativeClosureRunner runner(
                worker,
                {
                    "failure-reporting",
                    {{"U", Element::float64, 3}},
                    {{"nut", Element::float64, 1}},
                },
                bypass,
                kernel);
            static_cast<void>(runner.run_one());
        } catch (...) {
            workerFailure = std::current_exception();
        }
    });

    static_cast<void>(client.connect_session({
        9101,
        foamnordic::fjord::Capability::uds,
        0,
        1,
        4096,
    }));
    const auto unexpected = scalar_field("p", {1.0, 2.0}, 12, 0.3);
    client.send(unexpected.view());

    const auto response = client.receive_message();
    require(
        response.kind == foamnordic::fjord::RuneKind::error
            && response.exchange_index == 12,
        "Worker failure was not reported to the active solver exchange.");
    workerThread.join();
    require(
        workerFailure != nullptr,
        "Malformed closure input did not fail the native worker.");
}

}  // namespace

int main() {
    test_native_combustion_exchange(false);
    test_native_combustion_exchange(true);
    test_scaled_affine_exchange(false);
    test_scaled_affine_exchange(true);
    test_worker_failure_is_reported_to_solver(false);
    test_worker_failure_is_reported_to_solver(true);
}
