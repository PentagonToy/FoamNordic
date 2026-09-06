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
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/runtime/executor.hpp"

namespace foamnordic::inference {
class ForestKernel final : public PackedModelKernel {
public:
    explicit ForestKernel(std::span<const std::byte> data, std::uint32_t threads = 1);
    BackendCapabilities capabilities() const noexcept override { return {false, true, true}; }
    fjord::Tensor evaluate(fjord::TensorView input, std::uint64_t index,
                           double time, std::uint32_t rank = 0) override;
private:
    struct Node { std::int64_t left, right, feature; double threshold, value; bool missing_left; };
    std::vector<Node> nodes_;
    std::vector<std::uint64_t> roots_;
    std::uint64_t width_;
    std::uint32_t threads_;
    FnoExecutor executor_;
};
}
