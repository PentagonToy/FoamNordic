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

#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

#include "foamnordic/fjord/harbor.hpp"
#include "foamnordic/fjord/shm_channel.hpp"

namespace {

using ChannelPair =
    std::pair<std::unique_ptr<foamnordic::fjord::FjordChannel>,
              std::unique_ptr<foamnordic::fjord::FjordChannel>>;

constexpr std::size_t exchanges = 10'000;
constexpr std::size_t cells = 400;

struct Result {
    std::string_view channel;
    double elapsed;
    double exchanges_per_second;
    double payload_mib_per_second;
};

void serve(foamnordic::fjord::Harbor& worker) {
    std::array<double, cells> reaction_rate{};
    for (std::size_t index = 0; index < exchanges; ++index) {
        std::uint32_t prepared = 0;
        while (true) {
            const auto message = worker.receive_message();
            if (message.kind == foamnordic::fjord::RuneKind::tensor) {
                ++prepared;
                continue;
            }
            if (message.kind != foamnordic::fjord::RuneKind::complete
                || message.exchange_index != index
                || message.tensor_count != prepared) {
                throw std::runtime_error("Atomic benchmark exchange is incomplete.");
            }
            break;
        }
        const foamnordic::fjord::TensorView output{
            "omega_c",
            foamnordic::fjord::Element::float64,
            {cells},
            foamnordic::fjord::as_bytes(std::span(reaction_rate)),
            index,
            static_cast<double>(index) * 0.001,
        };
        worker.publish(index, std::span(&output, 1));
    }
}

Result exercise(std::string_view name, foamnordic::fjord::Harbor& client) {
    std::array<double, cells> progress{};
    std::array<double, cells> variance{};
    std::array<double, cells> temperature{};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < exchanges; ++index) {
        const auto time = static_cast<double>(index) * 0.001;
        const std::array inputs{
            foamnordic::fjord::TensorView{
                "c_tilde", foamnordic::fjord::Element::float64, {cells},
                foamnordic::fjord::as_bytes(std::span(progress)), index, time},
            foamnordic::fjord::TensorView{
                "c_var", foamnordic::fjord::Element::float64, {cells},
                foamnordic::fjord::as_bytes(std::span(variance)), index, time},
            foamnordic::fjord::TensorView{
                "T_tilde", foamnordic::fjord::Element::float64, {cells},
                foamnordic::fjord::as_bytes(std::span(temperature)), index, time},
        };
        client.publish(index, inputs);
        const auto output = client.receive_message();
        const auto commit = client.receive_message();
        if (output.kind != foamnordic::fjord::RuneKind::tensor
            || commit.kind != foamnordic::fjord::RuneKind::complete
            || commit.tensor_count != 1) {
            throw std::runtime_error("Atomic benchmark response is incomplete.");
        }
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const auto bytes_per_exchange =
        (progress.size() + variance.size() + temperature.size() + cells)
        * sizeof(double);
    return {
        name,
        elapsed,
        exchanges / elapsed,
        (exchanges * bytes_per_exchange) / (elapsed * 1024.0 * 1024.0),
    };
}

Result benchmark(std::string_view name, ChannelPair channels) {
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(30'000),
    };
    foamnordic::fjord::Harbor client(std::move(channels.first), options);
    foamnordic::fjord::Harbor worker(std::move(channels.second), options);
    std::thread closure([&worker] { serve(worker); });
    const auto result = exercise(name, client);
    closure.join();
    return result;
}

Result benchmark_process_shm() {
    const auto name = std::string("/foamnordic-benchmark-")
                      + std::to_string(static_cast<long long>(::getpid()));
    auto wake_channels = foamnordic::fjord::local_channel_pair();
    auto parent_channel = foamnordic::fjord::create_shared_memory_channel(
        name, std::move(wake_channels.first));
    const auto child = ::fork();
    if (child < 0) {
        throw std::runtime_error("Could not fork the SHM benchmark worker.");
    }
    if (child == 0) {
        try {
            const foamnordic::fjord::HarborOptions options{
                foamnordic::fjord::HandshakeMode::disabled,
                std::chrono::milliseconds(30'000),
            };
            foamnordic::fjord::Harbor worker(
                foamnordic::fjord::connect_shared_memory_channel(
                    name, std::move(wake_channels.second)),
                options);
            serve(worker);
            _exit(0);
        } catch (...) {
            _exit(2);
        }
    }
    wake_channels.second.reset();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(30'000),
    };
    foamnordic::fjord::Harbor client(std::move(parent_channel), options);
    const auto result = exercise("SHM/process", client);
    int status = 0;
    if (::waitpid(child, &status, 0) != child
        || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("The SHM benchmark worker failed.");
    }
    return result;
}

}  // namespace

int main() {
    const std::array results{
        benchmark("UDS", foamnordic::fjord::local_channel_pair()),
        benchmark("SHM", foamnordic::fjord::shared_memory_channel_pair()),
        benchmark_process_shm(),
    };
    std::cout << "FoamNordic atomic closure exchange\n\n"
              << std::left << std::setw(14) << "Channel"
              << std::right << std::setw(14) << "Elapsed (s)"
              << std::setw(18) << "Exchanges/s"
              << std::setw(18) << "Payload MiB/s" << '\n';
    for (const auto& result : results) {
        std::cout << std::left << std::setw(14) << result.channel
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << result.elapsed
                  << std::setprecision(0) << std::setw(18)
                  << result.exchanges_per_second
                  << std::setprecision(1) << std::setw(18)
                  << result.payload_mib_per_second << '\n';
    }
}
