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

#include <chrono>
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
      capabilities_(kernel.capabilities()),
      profile_every_(resources.profile_every) {
    artifact_.validate();
}

TensorMap ArtifactModelKernel::evaluate(
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t rank) {
    const auto sampled = profile_every_ != 0
                         && profile_sequence_.fetch_add(
                                1, std::memory_order_relaxed)
                                % profile_every_
                                == 0;
    const auto admission_started = sampled ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    auto context = contexts_.acquire(
        checked_working_bytes(smedja_, active_cells.size()));
    const auto packing_started = sampled
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
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
        artifact_.input_scaler->transform(mutable_view.model_view());
    }
    const auto inference_started = sampled
                                       ? std::chrono::steady_clock::now()
                                       : packing_started;
    if (active_cells.empty()) {
        auto outputs = smedja_.unpack({
            "bindnordic.predictions",
            smedja_.output_element(),
            {0, smedja_.output_width()},
            {},
            exchange_index,
            physical_time,
        });
        if (sampled) {
            const auto finished = std::chrono::steady_clock::now();
            admission_nanoseconds_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    packing_started - admission_started).count(),
                std::memory_order_relaxed);
            packing_nanoseconds_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    inference_started - packing_started).count(),
                std::memory_order_relaxed);
            output_nanoseconds_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    finished - inference_started).count(),
                std::memory_order_relaxed);
            profile_samples_.fetch_add(1, std::memory_order_relaxed);
        }
        return outputs;
    }
    bindnordic::Tensor model_predictions;
    auto backend_started = inference_started;
    if (capabilities_.concurrent_evaluation) {
        model_predictions = kernel_.evaluate(
            packed.view().model_view(), exchange_index, physical_time, rank);
    } else {
        auto backend_lease = backend_gate_.acquire(0);
        if (sampled) {
            backend_started = std::chrono::steady_clock::now();
        }
        model_predictions = kernel_.evaluate(
            packed.view().model_view(), exchange_index, physical_time, rank);
    }
    if (model_predictions.shape.empty()
        || model_predictions.shape.front() != active_cells.size()) {
        throw std::invalid_argument("Packed model output does not match its contract.");
    }
    auto predictions = fjord::Tensor{
        std::move(model_predictions.name),
        model_predictions.element,
        std::move(model_predictions.shape),
        std::move(model_predictions.bytes),
        exchange_index,
        physical_time,
        packed.solver_time_index,
    };
    const auto output_started = sampled
                                    ? std::chrono::steady_clock::now()
                                    : inference_started;
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
        artifact_.output_scaler->inverse_transform(mutable_view.model_view());
    }
    auto outputs = smedja_.unpack(std::move(predictions));
    if (sampled) {
        admission_nanoseconds_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                packing_started - admission_started).count(), std::memory_order_relaxed);
        backend_wait_nanoseconds_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                backend_started - inference_started).count(), std::memory_order_relaxed);
        const auto finished = std::chrono::steady_clock::now();
        packing_nanoseconds_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                inference_started - packing_started).count(),
            std::memory_order_relaxed);
        inference_nanoseconds_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                output_started - backend_started).count(),
            std::memory_order_relaxed);
        output_nanoseconds_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finished - output_started).count(),
            std::memory_order_relaxed);
        profile_samples_.fetch_add(1, std::memory_order_relaxed);
    }
    return outputs;
}

const ModelArtifact& ArtifactModelKernel::artifact() const noexcept { return artifact_; }

bool ArtifactModelKernel::owns_backend_synchronization() const noexcept {
    return true;
}

ExecutionSnapshot ArtifactModelKernel::execution_snapshot() const {
    auto snapshot = contexts_.snapshot();
    snapshot.profile_samples = profile_samples_.load(std::memory_order_relaxed);
    snapshot.packing_nanoseconds =
        packing_nanoseconds_.load(std::memory_order_relaxed);
    snapshot.inference_nanoseconds =
        inference_nanoseconds_.load(std::memory_order_relaxed);
    snapshot.output_nanoseconds =
        output_nanoseconds_.load(std::memory_order_relaxed);
    snapshot.admission_nanoseconds = admission_nanoseconds_.load(std::memory_order_relaxed);
    snapshot.backend_wait_nanoseconds = backend_wait_nanoseconds_.load(std::memory_order_relaxed);
    return snapshot;
}

void ArtifactModelKernel::cancel_pending() noexcept {
    contexts_.cancel();
    backend_gate_.cancel();
}

}  // namespace foamnordic::inference
