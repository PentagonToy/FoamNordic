#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bindnordic/kernel.hpp"

namespace bindnordic {

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
    std::vector<std::vector<double>> axes_;
    std::vector<bool> clamp_;
    std::vector<double> values_;
};

}  // namespace bindnordic
