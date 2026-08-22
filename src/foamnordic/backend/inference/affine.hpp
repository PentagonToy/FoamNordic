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

#pragma once

#include <cstdint>
#include <vector>

#include "foamnordic/backend/inference/model.hpp"

namespace foamnordic::closure {

class DenseAffineKernel final : public PackedModelKernel {
public:
    DenseAffineKernel(
        std::uint64_t input_features,
        std::uint64_t output_features,
        std::vector<double> weights,
        std::vector<double> bias);

    [[nodiscard]] fjord::Tensor evaluate(
        fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time) override;

    [[nodiscard]] std::uint64_t input_features() const noexcept;
    [[nodiscard]] std::uint64_t output_features() const noexcept;

private:
    std::uint64_t input_features_;
    std::uint64_t output_features_;
    std::vector<double> weights_;
    std::vector<double> bias_;
};

}  // namespace foamnordic::closure
