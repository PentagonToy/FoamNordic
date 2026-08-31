#include <chrono>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "foamnordic/backend/adapter/field.hpp"

int main() {
    constexpr std::uint64_t cells = 1'000'000;
    constexpr std::size_t repetitions = 20;
    std::vector<double> velocity(cells * 3, 1.0);
    foamnordic::fjord::MutableTensorView field{
        "U",
        foamnordic::fjord::Element::float64,
        {cells, 3},
        foamnordic::fjord::as_writable_bytes(std::span(velocity)),
        1,
        0.1,
    };
    foamnordic::adapter::FieldTransform transform{
        {1.01, 0.99, 1.0},
        {0.0, 0.01, 0.0},
        -10.0,
        10.0,
    };

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < repetitions; ++index) {
        transform.apply_in_place(field);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const auto bytes = static_cast<double>(velocity.size() * sizeof(double) * repetitions);
    const auto summary = foamnordic::adapter::statistics(field.read_only());

    std::cout << "Cells: " << cells << '\n'
              << "Fused passes: " << repetitions << '\n'
              << "Elapsed: " << elapsed << " s\n"
              << "Field throughput: " << bytes / elapsed / (1024.0 * 1024.0)
              << " MiB/s\n"
              << "Final range: [" << summary.minimum << ", " << summary.maximum << "]\n";
}
