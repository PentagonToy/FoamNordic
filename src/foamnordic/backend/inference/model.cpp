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
#include <limits>
#include <stdexcept>
#include <utility>

namespace foamnordic::inference {
namespace {

std::uint64_t checked_tensor_bytes(
    std::size_t rows,
    std::uint64_t width,
    fjord::Element element) {
    const auto item_bytes = static_cast<std::uint64_t>(
        fjord::element_size(element));
    const auto row_count = static_cast<std::uint64_t>(rows);
    if (width != 0
        && row_count > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::overflow_error("Model execution row count overflowed.");
    }
    const auto elements = row_count * width;
    if (item_bytes != 0
        && elements > std::numeric_limits<std::uint64_t>::max() / item_bytes) {
        throw std::overflow_error("Model execution byte count overflowed.");
    }
    return elements * item_bytes;
}

std::uint64_t checked_working_bytes(
    const Smedja& smedja,
    std::size_t rows) {
    const auto input = checked_tensor_bytes(
        rows, smedja.input_width(), smedja.input_element());
    const auto output = checked_tensor_bytes(
        rows, smedja.output_width(), smedja.output_element());
    if (output > (std::numeric_limits<std::uint64_t>::max() - input) / 2) {
        throw std::overflow_error("Model execution working set overflowed.");
    }
    // Predictions and committed named outputs may coexist during unpacking.
    return input + 2 * output;
}

}  // namespace

ArtifactModelKernel::ArtifactModelKernel(
    ModelArtifact artifact,
    PackedModelKernel& kernel,
    ExecutionResources resources)
    : artifact_(std::move(artifact)),
      kernel_(kernel),
      smedja_(artifact_.contract),
      contexts_(resources),
      capabilities_(kernel.capabilities()) {
    artifact_.validate();
}

TensorMap ArtifactModelKernel::evaluate(
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t rank) {
    auto context = contexts_.acquire(
        checked_working_bytes(smedja_, active_cells.size()));
    auto& packed = smedja_.pack_into(
        context.workspace(), inputs, active_cells, exchange_index, physical_time);
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
    if (capabilities_.concurrent_evaluation) {
        predictions = kernel_.evaluate(
            packed.view(), exchange_index, physical_time, rank);
    } else {
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

ExecutionSnapshot ArtifactModelKernel::execution_snapshot() const {
    return contexts_.snapshot();
}

void ArtifactModelKernel::cancel_pending() noexcept { contexts_.cancel(); }

}  // namespace foamnordic::inference
