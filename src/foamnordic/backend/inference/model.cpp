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

#include "foamnordic/backend/inference/model.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace foamnordic::inference {

ArtifactModelKernel::ArtifactModelKernel(ModelArtifact artifact, PackedModelKernel& kernel)
    : artifact_(std::move(artifact)), kernel_(kernel), smedja_(artifact_.contract) {
    artifact_.validate();
}

TensorMap ArtifactModelKernel::evaluate(
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t rank) {
    // InferenceRunner threads persist for the worker lifetime. This workspace
    // therefore reuses capacity per runner without sharing OpenFOAM addresses
    // or mutable tensor data between ranks.
    thread_local SmedjaWorkspace workspace;
    auto& packed = smedja_.pack_into(
        workspace, inputs, active_cells, exchange_index, physical_time);
    if (artifact_.input_scaler) {
        auto mutable_view = fjord::MutableTensorView{
            packed.name,
            packed.element,
            packed.shape,
            packed.bytes,
            packed.time_index,
            packed.physical_time,
            packed.solver_time_index,
        };
        artifact_.input_scaler->transform(mutable_view);
    }
    if (active_cells.empty()) {
        return smedja_.unpack({
            "foamnordic.predictions",
            smedja_.output_element(),
            {0, smedja_.output_width()},
            {},
            exchange_index,
            physical_time,
        });
    }
    fjord::Tensor predictions;
    {
        std::scoped_lock lock(backend_mutex_);
        predictions = kernel_.evaluate(
            packed.view(), exchange_index, physical_time, rank);
    }
    if (predictions.time_index != exchange_index
        || std::abs(predictions.physical_time - physical_time) > 1.0e-12
        || predictions.shape.empty() || predictions.shape.front() != active_cells.size()) {
        throw std::invalid_argument("Packed model output does not match its exchange.");
    }
    if (artifact_.output_scaler) {
        auto mutable_view = fjord::MutableTensorView{
            predictions.name,
            predictions.element,
            predictions.shape,
            predictions.bytes,
            predictions.time_index,
            predictions.physical_time,
            predictions.solver_time_index,
        };
        artifact_.output_scaler->inverse_transform(mutable_view);
    }
    return smedja_.unpack(std::move(predictions));
}

const ModelArtifact& ArtifactModelKernel::artifact() const noexcept { return artifact_; }

bool ArtifactModelKernel::owns_backend_synchronization() const noexcept {
    return true;
}

}  // namespace foamnordic::inference
