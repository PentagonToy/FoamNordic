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

#include <mutex>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/smedja.hpp"
#include "foamnordic/backend/inference/runner.hpp"

namespace foamnordic::inference {

class PackedModelKernel {
public:
    virtual ~PackedModelKernel() = default;

    [[nodiscard]] virtual fjord::Tensor evaluate(
        fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) = 0;
};

class ArtifactModelKernel final : public ModelKernel {
public:
    ArtifactModelKernel(ModelArtifact artifact, PackedModelKernel& kernel);

    [[nodiscard]] TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) override;

    [[nodiscard]] bool owns_backend_synchronization() const noexcept override;

    [[nodiscard]] const ModelArtifact& artifact() const noexcept;

private:
    ModelArtifact artifact_;
    PackedModelKernel& kernel_;
    Smedja smedja_;
    std::mutex backend_mutex_;
};

}  // namespace foamnordic::inference
