// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * BindNordic
 */
// clang-format on

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bindnordic/kernel.hpp"

namespace bindnordic {

struct TableAxis {
    std::vector<double> points;
    double start{0.0};
    double step{0.0};
    std::size_t count{0};
    bool uniform{false};
};

class TableKernel final : public PackedModelKernel {
public:
    TableKernel(
        std::span<const std::byte> payload,
        std::uint64_t input_features,
        std::uint64_t output_features,
        Element element);

    [[nodiscard]] BackendCapabilities capabilities() const noexcept override;
    [[nodiscard]] Tensor evaluate(
        TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) override;

private:
    Element element_;
    std::uint64_t output_features_;
    std::vector<TableAxis> axes_;
    std::vector<bool> clamp_;
    std::vector<double> values_;
};

}  // namespace bindnordic
