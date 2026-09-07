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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#endif

#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace {

constexpr std::uint64_t features = 10;
constexpr std::uint64_t small_rows = 1024;
constexpr std::uint64_t large_rows = 1024 * 1024;
constexpr std::uint64_t maximum_payload = 128ULL * 1024ULL * 1024ULL;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::uint64_t resident_kib(pid_t process) {
#ifdef __APPLE__
    proc_taskinfo information{};
    const auto size = ::proc_pidinfo(
        process,
        PROC_PIDTASKINFO,
        0,
        &information,
        sizeof(information));
    return size == sizeof(information) ? information.pti_resident_size / 1024 : 0;
#else
    std::ifstream status("/proc/" + std::to_string(process) + "/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            std::uint64_t value{};
            std::string unit;
            status >> value >> unit;
            return value;
        }
        std::string remainder;
        std::getline(status, remainder);
    }
    return 0;
#endif
}

[[nodiscard]] foamnordic::fjord::Tensor input_tensor(
    std::uint64_t rows,
    std::uint64_t exchange_index) {
    std::vector<double> values(rows * features);
    for (std::uint64_t row = 0; row < rows; ++row) {
        values[row * features] = static_cast<double>(row % 1024) * 1.0e-3;
    }
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        "features",
        foamnordic::fjord::Element::float64,
        {rows, features},
        std::move(bytes),
        exchange_index,
        static_cast<double>(exchange_index),
        exchange_index,
    };
}

void exchange(
    foamnordic::fjord::Harbor& client,
    std::uint64_t rows,
    std::uint64_t exchange_index) {
    auto input = input_tensor(rows, exchange_index);
    const auto view = input.view();
    client.publish(exchange_index, std::span(&view, 1));
    auto output = client.receive_message();
    require(
        output.kind == foamnordic::fjord::RuneKind::tensor
            && output.tensor.has_value(),
        "ModelHost did not publish an output tensor.");
    require(
        output.tensor->name == "prediction"
            && output.tensor->bytes.size() == rows * sizeof(double),
        "ModelHost output contract is incorrect.");
    std::uint64_t completed{};
    require(
        client.receive_control(&completed)
                == foamnordic::fjord::RuneKind::complete
            && completed == exchange_index,
        "ModelHost did not publish an atomic completion.");
}

[[noreturn]] void run_worker(const std::filesystem::path& socket_path) {
    try {
        const foamnordic::inference::ProgramContract contract{
            "resident-memory",
            {{"features", foamnordic::fjord::Element::float64, features}},
            {{"prediction", foamnordic::fjord::Element::float64, 1}},
        };
        std::vector<double> weights(features, 0.1);
        foamnordic::inference::DenseAffineKernel affine(
            features, 1, std::move(weights), {0.0});
        const foamnordic::inference::ModelArtifact artifact{
            foamnordic::inference::ModelFormat::onnx,
            "resident-memory-reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        };
        foamnordic::inference::ArtifactModelKernel kernel(
            artifact,
            affine,
            {
                1,
                256ULL * 1024ULL * 1024ULL,
                1,
            });
        foamnordic::inference::EvaluateAllCells policy;
        foamnordic::inference::WorkerOptions options;
        options.connections = 1;
        options.model_threads = 1;
        options.maximum_payload = maximum_payload;
        options.maximum_receive_bytes = 256ULL * 1024ULL * 1024ULL;
        options.maximum_working_bytes = 256ULL * 1024ULL * 1024ULL;
        options.profile_every = 1;
        foamnordic::inference::ModelWorker worker(
            foamnordic::fjord::FjordAddress::local(socket_path.string()),
            artifact,
            policy,
            kernel,
            options);
        worker.run();
        std::_Exit(0);
    } catch (const std::exception& error) {
        std::cerr << "ModelHost child failed: " << error.what() << std::endl;
        std::_Exit(1);
    }
}

}  // namespace

int main() {
    const auto socket_path = std::filesystem::temp_directory_path()
                             / ("foamnordic-memory-"
                                + std::to_string(std::random_device{}()) + ".sock");
    const auto child = ::fork();
    if (child < 0) {
        throw std::runtime_error("Could not fork the ModelHost benchmark.");
    }
    if (child == 0) {
        run_worker(socket_path);
    }

    std::atomic<bool> sampling{true};
    std::atomic<std::uint64_t> peak_kib{0};
    const auto observe_rss = [&]() {
        const auto current = resident_kib(child);
        auto peak = peak_kib.load();
        while (current > peak &&
               !peak_kib.compare_exchange_weak(peak, current)) {
        }
        return current;
    };
    std::thread sampler([&] {
        while (sampling.load()) {
            observe_rss();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    try {
        for (int attempt = 0; attempt < 400
             && !std::filesystem::exists(socket_path); ++attempt) {
            int status{};
            if (::waitpid(child, &status, WNOHANG) == child) {
                throw std::runtime_error("ModelHost exited before readiness.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        require(
            std::filesystem::exists(socket_path),
            "ModelHost readiness timed out.");

        foamnordic::fjord::Harbor client(
            foamnordic::fjord::connect(
                foamnordic::fjord::FjordAddress::local(socket_path.string())));
        const auto session = client.connect_session({
            1,
            foamnordic::fjord::Capability::uds
                | foamnordic::fjord::Capability::shm,
            0,
            1,
            maximum_payload,
        });
        if (foamnordic::fjord::any(
                session.capabilities & foamnordic::fjord::Capability::shm)) {
            client.accept_shared_memory();
        }
        std::cout << "Data plane: "
                  << (foamnordic::fjord::any(
                          session.capabilities
                          & foamnordic::fjord::Capability::shm)
                          ? "SHM"
                          : "UDS")
                  << '\n';
        std::cout << "RSS connected: " << observe_rss() << " KiB\n";

        std::uint64_t exchange_index = 0;
        for (int repetition = 0; repetition < 20; ++repetition) {
            exchange(client, small_rows, exchange_index++);
        }
        std::cout << "RSS after small exchanges: " << observe_rss()
                  << " KiB\n";

        for (int repetition = 0; repetition < 3; ++repetition) {
            exchange(client, large_rows, exchange_index++);
        }
        std::cout << "RSS after large exchanges: " << observe_rss()
                  << " KiB\n";

        for (int repetition = 0; repetition < 20; ++repetition) {
            exchange(client, small_rows, exchange_index++);
        }
        std::cout << "RSS after returning to small exchanges: "
                  << observe_rss() << " KiB\n";
        client.shutdown();

        int status{};
        require(::waitpid(child, &status, 0) == child, "Could not reap ModelHost.");
        require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "ModelHost failed.");
        sampling.store(false);
        sampler.join();
        std::cout << "Peak sampled ModelHost RSS: " << peak_kib.load() << " KiB\n";
        std::cout << "Small rows: " << small_rows << '\n';
        std::cout << "Large rows: " << large_rows << '\n';
        std::cout << "Resident memory benchmark: PASS" << std::endl;
    } catch (...) {
        sampling.store(false);
        sampler.join();
        ::kill(child, SIGTERM);
        static_cast<void>(::waitpid(child, nullptr, 0));
        std::filesystem::remove(socket_path);
        throw;
    }
}
