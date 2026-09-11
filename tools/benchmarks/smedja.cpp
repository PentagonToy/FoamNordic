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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "foamnordic/backend/inference/smedja.hpp"

namespace {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void consume(const float* data, std::size_t size) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(data), "r"(size) : "memory");
#else
    volatile auto value = data[size == 0 ? 0 : size - 1];
    static_cast<void>(value);
#endif
}

foamnordic::fjord::Tensor tensor(std::string name, const std::vector<double>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        foamnordic::fjord::Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
    };
}

}  // namespace

int main() {
    constexpr std::uint64_t cells = 1'000'000;
    constexpr std::size_t repetitions = 200;
    foamnordic::inference::Smedja smedja({
        "dtype-pack",
        {
            {"c", foamnordic::fjord::Element::float32, 1},
            {"c_var", foamnordic::fjord::Element::float32, 1},
            {"T", foamnordic::fjord::Element::float32, 1},
        },
        {{"omega", foamnordic::fjord::Element::float32, 1}},
    });
    foamnordic::inference::TensorMap inputs;
    inputs.emplace("c", tensor("c", std::vector<double>(cells, 0.5)));
    inputs.emplace("c_var", tensor("c_var", std::vector<double>(cells, 0.01)));
    inputs.emplace("T", tensor("T", std::vector<double>(cells, 900.0)));
    std::vector<std::uint64_t> active(cells);
    std::iota(active.begin(), active.end(), 0);
    foamnordic::inference::SmedjaWorkspace workspace;

    const auto& c = inputs.at("c");
    const auto& c_var = inputs.at("c_var");
    const auto& temperature = inputs.at("T");
    const auto* c_values = reinterpret_cast<const double*>(c.bytes.data());
    const auto* variance_values =
        reinterpret_cast<const double*>(c_var.bytes.data());
    const auto* temperature_values =
        reinterpret_cast<const double*>(temperature.bytes.data());
    std::vector<float> converted_c(cells);
    std::vector<float> converted_variance(cells);
    std::vector<float> converted_temperature(cells);
    std::vector<float> two_pass(cells * 3);
    std::vector<float> fused(cells * 3);

    const auto baseline_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        for (std::uint64_t cell = 0; cell < cells; ++cell) {
            converted_c[cell] = static_cast<float>(c_values[cell]);
            converted_variance[cell] = static_cast<float>(variance_values[cell]);
            converted_temperature[cell] = static_cast<float>(temperature_values[cell]);
        }
        for (std::uint64_t cell = 0; cell < cells; ++cell) {
            two_pass[cell * 3] = converted_c[cell];
            two_pass[cell * 3 + 1] = converted_variance[cell];
            two_pass[cell * 3 + 2] = converted_temperature[cell];
        }
        consume(two_pass.data(), two_pass.size());
    }
    const auto baseline_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - baseline_start).count();

    const auto fused_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        for (std::uint64_t cell = 0; cell < cells; ++cell) {
            fused[cell * 3] = static_cast<float>(c_values[cell]);
            fused[cell * 3 + 1] = static_cast<float>(variance_values[cell]);
            fused[cell * 3 + 2] = static_cast<float>(temperature_values[cell]);
        }
        consume(fused.data(), fused.size());
    }
    const auto fused_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fused_start).count();

    const auto scan_start = std::chrono::steady_clock::now();
    std::uint64_t scan_checksum = 0;
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        std::uint64_t maximum = 0;
        bool contiguous = true;
        for (std::size_t row = 0; row < active.size(); ++row) {
            maximum = std::max(maximum, active[row]);
            contiguous = contiguous && active[row] >= active.front()
                && active[row] - active.front() == row;
        }
        scan_checksum += maximum + static_cast<std::uint64_t>(contiguous);
    }
    const auto scan_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scan_start).count();

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        const auto& packed = smedja.pack_validated_into(
            workspace, inputs, active, iteration + 1, 0.001 * iteration);
        consume(reinterpret_cast<const float*>(packed.bytes.data()), cells * 3);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const auto source_bytes = static_cast<double>(
        cells * 3 * sizeof(double) * repetitions);
    std::cout << "Cells: " << cells << '\n'
              << "Fused packs: " << repetitions << '\n'
              << "Two-pass elapsed: " << baseline_elapsed << " s\n"
              << "Ideal fused elapsed: " << fused_elapsed << " s\n"
              << "Index scan elapsed: " << scan_elapsed << " s\n"
              << "Smedja elapsed: " << elapsed << " s\n"
              << "Smedja / ideal fused: " << elapsed / fused_elapsed << "x\n"
              << "Source throughput: "
              << source_bytes / elapsed / (1024.0 * 1024.0) << " MiB/s\n"
              << "Checksum: " << scan_checksum + two_pass[0] + fused[0] << '\n';
}
