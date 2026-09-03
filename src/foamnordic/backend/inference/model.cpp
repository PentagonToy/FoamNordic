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
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace foamnordic::inference {
namespace {

std::uint64_t features(const std::vector<FieldContract>& fields) {
    std::uint64_t count = 0;
    for (const auto& field : fields) {
        if (count > std::numeric_limits<std::uint64_t>::max() - field.components) {
            throw std::overflow_error("Packed model feature count overflowed.");
        }
        count += field.components;
    }
    return count;
}

fjord::Element common_element(const std::vector<FieldContract>& fields) {
    const auto element = fields.front().element;
    for (const auto& field : fields) {
        if (field.element != element) {
            throw std::invalid_argument(
                "Packed model fields must use one floating-point element type.");
        }
    }
    if (element != fjord::Element::float32 && element != fjord::Element::float64) {
        throw std::invalid_argument("Packed model fields must be floating point.");
    }
    return element;
}

}  // namespace

ArtifactModelKernel::ArtifactModelKernel(ModelArtifact artifact, PackedModelKernel& kernel)
    : artifact_(std::move(artifact)), kernel_(kernel) {
    artifact_.validate();
    static_cast<void>(common_element(artifact_.contract.inputs));
    static_cast<void>(common_element(artifact_.contract.outputs));
}

fjord::Tensor ArtifactModelKernel::pack_inputs(
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time) const {
    const auto element = common_element(artifact_.contract.inputs);
    const auto width = features(artifact_.contract.inputs);
    const auto item_bytes = fjord::element_size(element);
    std::vector<std::byte> bytes(active_cells.size() * width * item_bytes);

    for (std::size_t row = 0; row < active_cells.size(); ++row) {
        std::uint64_t feature_offset = 0;
        for (const auto& field : artifact_.contract.inputs) {
            const auto& tensor = inputs.at(field.name);
            if (active_cells[row] >= tensor.shape.front()) {
                throw std::out_of_range("Active cell exceeds a model input field.");
            }
            const auto source_offset = active_cells[row] * field.components * item_bytes;
            const auto destination_offset =
                (row * width + feature_offset) * item_bytes;
            const auto copy_bytes = field.components * item_bytes;
            std::memcpy(
                bytes.data() + destination_offset,
                tensor.bytes.data() + source_offset,
                copy_bytes);
            feature_offset += field.components;
        }
    }

    fjord::Tensor packed{
        "foamnordic.features",
        element,
        {static_cast<std::uint64_t>(active_cells.size()), width},
        std::move(bytes),
        exchange_index,
        physical_time,
    };
    if (artifact_.input_scaler) {
        auto mutable_view = fjord::MutableTensorView{
            packed.name,
            packed.element,
            packed.shape,
            packed.bytes,
            packed.time_index,
            packed.physical_time,
        };
        artifact_.input_scaler->transform(mutable_view);
    }
    return packed;
}

TensorMap ArtifactModelKernel::unpack_outputs(fjord::Tensor packed) const {
    packed.view().validate();
    const auto element = common_element(artifact_.contract.outputs);
    const auto width = features(artifact_.contract.outputs);
    if (packed.element != element || packed.shape.size() != 2
        || packed.shape[1] != width) {
        throw std::invalid_argument("Packed model output metadata is incorrect.");
    }
    if (artifact_.output_scaler) {
        auto mutable_view = fjord::MutableTensorView{
            packed.name,
            packed.element,
            packed.shape,
            packed.bytes,
            packed.time_index,
            packed.physical_time,
        };
        artifact_.output_scaler->inverse_transform(mutable_view);
    }

    TensorMap outputs;
    const auto item_bytes = fjord::element_size(element);
    std::uint64_t feature_offset = 0;
    for (const auto& field : artifact_.contract.outputs) {
        std::vector<std::byte> bytes(
            packed.shape[0] * field.components * item_bytes);
        for (std::uint64_t row = 0; row < packed.shape[0]; ++row) {
            std::memcpy(
                bytes.data() + row * field.components * item_bytes,
                packed.bytes.data() + (row * width + feature_offset) * item_bytes,
                field.components * item_bytes);
        }
        std::vector<std::uint64_t> shape{packed.shape[0]};
        if (field.components != 1) {
            shape.push_back(field.components);
        }
        outputs.emplace(
            field.name,
            fjord::Tensor{
                field.name,
                element,
                std::move(shape),
                std::move(bytes),
                packed.time_index,
                packed.physical_time,
            });
        feature_offset += field.components;
    }
    return outputs;
}

TensorMap ArtifactModelKernel::evaluate(
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t rank) {
    auto packed = pack_inputs(inputs, active_cells, exchange_index, physical_time);
    if (active_cells.empty()) {
        const auto element = common_element(artifact_.contract.outputs);
        return unpack_outputs({
            "foamnordic.predictions",
            element,
            {0, features(artifact_.contract.outputs)},
            {},
            exchange_index,
            physical_time,
        });
    }
    auto predictions = kernel_.evaluate(
        packed.view(), exchange_index, physical_time, rank);
    if (predictions.time_index != exchange_index
        || std::abs(predictions.physical_time - physical_time) > 1.0e-12
        || predictions.shape.empty() || predictions.shape.front() != active_cells.size()) {
        throw std::invalid_argument("Packed model output does not match its exchange.");
    }
    return unpack_outputs(std::move(predictions));
}

const ModelArtifact& ArtifactModelKernel::artifact() const noexcept { return artifact_; }

}  // namespace foamnordic::inference
