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

#include "foamnordic/backend/inference/smedja.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace foamnordic::inference {
namespace {

std::size_t checked_bytes(std::uint64_t rows, std::uint64_t width, std::size_t item_bytes) {
    if (width != 0 && rows > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::overflow_error("Smedja tensor element count overflowed.");
    }
    const auto elements = rows * width;
    if (item_bytes != 0 && elements > std::numeric_limits<std::size_t>::max() / item_bytes) {
        throw std::overflow_error("Smedja tensor byte count overflowed.");
    }
    return static_cast<std::size_t>(elements) * item_bytes;
}

void copy_values(
    std::byte* destination,
    fjord::Element destination_element,
    const std::byte* source,
    fjord::Element source_element,
    std::uint64_t count) {
    if (destination_element == source_element) {
        std::memcpy(
            destination,
            source,
            static_cast<std::size_t>(count) * fjord::element_size(source_element));
        return;
    }
    if (destination_element == fjord::Element::float32
        && source_element == fjord::Element::float64) {
        auto* output = reinterpret_cast<float*>(destination);
        const auto* input = reinterpret_cast<const double*>(source);
        for (std::uint64_t index = 0; index < count; ++index) {
            output[index] = static_cast<float>(input[index]);
        }
        return;
    }
    if (destination_element == fjord::Element::float64
        && source_element == fjord::Element::float32) {
        auto* output = reinterpret_cast<double*>(destination);
        const auto* input = reinterpret_cast<const float*>(source);
        for (std::uint64_t index = 0; index < count; ++index) {
            output[index] = static_cast<double>(input[index]);
        }
        return;
    }
    throw std::invalid_argument("Smedja supports only floating-point input conversion.");
}

} // namespace

Smedja::Smedja(const ProgramContract& contract)
    : inputs_(compile(contract.inputs)), outputs_(compile(contract.outputs)) {}

std::size_t SmedjaWorkspace::retained_bytes() const noexcept {
    return input_.bytes.capacity();
}

Smedja::Layout Smedja::compile(const std::vector<FieldContract>& fields) {
    if (fields.empty()) {
        throw std::invalid_argument("Smedja layout requires at least one field.");
    }
    Layout layout;
    layout.element = fields.front().element;
    layout.fields.reserve(fields.size());
    for (const auto& field : fields) {
        field.validate();
        if (field.element != layout.element) {
            throw std::invalid_argument("Smedja fields must use one floating-point element type.");
        }
        if (field.element != fjord::Element::float32 && field.element != fjord::Element::float64) {
            throw std::invalid_argument("Smedja fields must be floating point.");
        }
        if (layout.width > std::numeric_limits<std::uint64_t>::max() - field.components) {
            throw std::overflow_error("Smedja feature count overflowed.");
        }
        layout.fields.push_back({
            field.name,
            field.element,
            field.components,
            layout.width,
        });
        layout.width += field.components;
    }
    return layout;
}

fjord::Tensor Smedja::pack(const TensorMap& inputs, const std::vector<std::uint64_t>& active_cells,
                           std::uint64_t exchange_index, double physical_time) const {
    SmedjaWorkspace workspace;
    auto& packed = pack_into(
        workspace, inputs, active_cells, exchange_index, physical_time);
    return std::move(packed);
}

fjord::Tensor& Smedja::pack_into(
    SmedjaWorkspace& workspace,
    const TensorMap& inputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t exchange_index,
    double physical_time) const {
    struct Binding {
        const FieldLayout* layout;
        const fjord::Tensor* tensor;
    };

    // These bindings are invocation-local by design.  They remove string-map
    // lookup from the cell loop without caching an OpenFOAM-owned address.
    std::vector<Binding> bindings;
    bindings.reserve(inputs_.fields.size());
    for (const auto& field : inputs_.fields) {
        const auto found = inputs.find(field.name);
        if (found == inputs.end()) {
            throw std::invalid_argument("Smedja input field is missing: " + field.name);
        }
        const auto& tensor = found->second;
        tensor.view().validate();
        if ((tensor.element != fjord::Element::float32
             && tensor.element != fjord::Element::float64)
            || tensor.shape.empty()) {
            throw std::invalid_argument("Smedja input metadata does not match its contract.");
        }
        const auto expected_dimensions = field.components == 1 ? 1U : 2U;
        if (tensor.shape.size() != expected_dimensions ||
            (field.components != 1 && tensor.shape[1] != field.components)) {
            throw std::invalid_argument("Smedja input shape does not match its contract.");
        }
        bindings.push_back({&field, &found->second});
    }

    bool contiguous_cells = true;
    if (!active_cells.empty()) {
        std::uint64_t maximum = 0;
        for (std::size_t row = 0; row < active_cells.size(); ++row) {
            maximum = std::max(maximum, active_cells[row]);
            contiguous_cells = contiguous_cells
                && active_cells[row] >= active_cells.front()
                && active_cells[row] - active_cells.front() == row;
        }
        for (const auto& binding : bindings) {
            if (maximum >= binding.tensor->shape.front()) {
                throw std::out_of_range("Active cell exceeds a Smedja input field.");
            }
        }
    }

    const auto item_bytes = fjord::element_size(inputs_.element);
    const auto byte_count = checked_bytes(
        active_cells.size(), inputs_.width, item_bytes);
    auto& bytes = workspace.input_.bytes;
    // Capacity is execution storage, not mesh state. Keep it for similarly
    // sized invocations but release a buffer more than four times the current
    // request so a temporary large topology does not pin memory indefinitely.
    if (byte_count != 0 && bytes.capacity() / byte_count > 4) {
        std::vector<std::byte>(byte_count).swap(bytes);
    } else {
        bytes.resize(byte_count);
    }
    // The common all-cell path is a contiguous range. Avoid re-reading its
    // index vector in the hot nested loop and copy a sole model input as one
    // block. Bypass policies with sparse cells retain the general gather path.
    if (contiguous_cells && !active_cells.empty() && bindings.size() == 1) {
        const auto& binding = bindings.front();
        const auto source_item_bytes = fjord::element_size(binding.tensor->element);
        const auto source_offset = active_cells.front()
            * binding.layout->components * source_item_bytes;
        copy_values(
            bytes.data(),
            inputs_.element,
            binding.tensor->bytes.data() + source_offset,
            binding.tensor->element,
            static_cast<std::uint64_t>(active_cells.size())
                * binding.layout->components);
    } else if (contiguous_cells && !active_cells.empty()) {
        const auto first_cell = active_cells.front();
        const auto three_scalar_double_to_float =
            bindings.size() == 3
            && inputs_.element == fjord::Element::float32
            && std::all_of(bindings.begin(), bindings.end(), [](const auto& binding) {
                   return binding.layout->components == 1
                       && binding.tensor->element == fjord::Element::float64;
               });
        if (three_scalar_double_to_float) {
            auto* packed = reinterpret_cast<float*>(bytes.data());
            const auto* first = reinterpret_cast<const double*>(
                bindings[0].tensor->bytes.data()) + first_cell;
            const auto* second = reinterpret_cast<const double*>(
                bindings[1].tensor->bytes.data()) + first_cell;
            const auto* third = reinterpret_cast<const double*>(
                bindings[2].tensor->bytes.data()) + first_cell;
            for (std::size_t row = 0; row < active_cells.size(); ++row) {
                packed[row * 3] = static_cast<float>(first[row]);
                packed[row * 3 + 1] = static_cast<float>(second[row]);
                packed[row * 3 + 2] = static_cast<float>(third[row]);
            }
        } else {
            for (const auto& binding : bindings) {
                const auto& field = *binding.layout;
                const auto source_item_bytes =
                    fjord::element_size(binding.tensor->element);
                const auto* source = binding.tensor->bytes.data()
                    + first_cell * field.components * source_item_bytes;
                auto* destination = bytes.data() + field.feature_offset * item_bytes;
                if (field.components == 1
                    && inputs_.element == fjord::Element::float32
                    && binding.tensor->element == fjord::Element::float64) {
                    const auto* values = reinterpret_cast<const double*>(source);
                    auto* packed = reinterpret_cast<float*>(destination);
                    for (std::size_t row = 0; row < active_cells.size(); ++row) {
                        packed[row * inputs_.width] = static_cast<float>(values[row]);
                    }
                } else if (field.components == 1
                           && inputs_.element == fjord::Element::float64
                           && binding.tensor->element == fjord::Element::float32) {
                    const auto* values = reinterpret_cast<const float*>(source);
                    auto* packed = reinterpret_cast<double*>(destination);
                    for (std::size_t row = 0; row < active_cells.size(); ++row) {
                        packed[row * inputs_.width] = static_cast<double>(values[row]);
                    }
                } else {
                    for (std::size_t row = 0; row < active_cells.size(); ++row) {
                        const auto source_offset =
                            row * field.components * source_item_bytes;
                        const auto destination_offset =
                            row * inputs_.width * item_bytes;
                        copy_values(
                            destination + destination_offset,
                            inputs_.element,
                            source + source_offset,
                            binding.tensor->element,
                            field.components);
                    }
                }
            }
        }
    } else {
        for (std::size_t row = 0; row < active_cells.size(); ++row) {
            for (const auto& binding : bindings) {
                const auto& field = *binding.layout;
                const auto source_item_bytes =
                    fjord::element_size(binding.tensor->element);
                const auto source_offset =
                    active_cells[row] * field.components * source_item_bytes;
                const auto destination_offset =
                    (row * inputs_.width + field.feature_offset) * item_bytes;
                copy_values(
                    bytes.data() + destination_offset,
                    inputs_.element,
                    binding.tensor->bytes.data() + source_offset,
                    binding.tensor->element,
                    field.components);
            }
        }
    }

    workspace.input_.name = "foamnordic.features";
    workspace.input_.element = inputs_.element;
    workspace.input_.shape = {
        static_cast<std::uint64_t>(active_cells.size()), inputs_.width};
    workspace.input_.time_index = exchange_index;
    workspace.input_.physical_time = physical_time;
    workspace.input_.solver_time_index = 0;
    return workspace.input_;
}

TensorMap Smedja::unpack(fjord::Tensor packed) const {
    packed.view().validate();
    if (packed.element != outputs_.element || packed.shape.size() != 2 ||
        packed.shape[1] != outputs_.width) {
        throw std::invalid_argument("Smedja output metadata does not match its contract.");
    }

    TensorMap outputs;
    if (outputs_.fields.size() == 1) {
        const auto& field = outputs_.fields.front();
        packed.name = field.name;
        packed.shape = {packed.shape.front()};
        if (field.components != 1) {
            packed.shape.push_back(field.components);
        }
        outputs.emplace(field.name, std::move(packed));
        return outputs;
    }

    const auto item_bytes = fjord::element_size(outputs_.element);
    const auto retained = std::max_element(
        outputs_.fields.begin(),
        outputs_.fields.end(),
        [](const auto& left, const auto& right) {
            return left.components < right.components;
        });
    for (auto field = outputs_.fields.begin(); field != outputs_.fields.end(); ++field) {
        if (field == retained) {
            continue;
        }
        std::vector<std::byte> bytes(
            checked_bytes(packed.shape[0], field->components, item_bytes));
        for (std::uint64_t row = 0; row < packed.shape[0]; ++row) {
            std::memcpy(bytes.data() + row * field->components * item_bytes,
                        packed.bytes.data() +
                            (row * outputs_.width + field->feature_offset) * item_bytes,
                        field->components * item_bytes);
        }
        std::vector<std::uint64_t> shape{packed.shape[0]};
        if (field->components != 1) {
            shape.push_back(field->components);
        }
        outputs.emplace(field->name, fjord::Tensor{
                                        field->name,
                                        outputs_.element,
                                        std::move(shape),
                                        std::move(bytes),
                                        packed.time_index,
                                        packed.physical_time,
                                        packed.solver_time_index,
                                    });
    }

    // Preserve the backend allocation for the widest output. Other outputs
    // are extracted first, then the retained field is compacted forward in
    // the packed buffer and transferred to its Tensor. This removes one
    // full-field allocation without exposing a strided or aliased view to the
    // atomic publication boundary.
    for (std::uint64_t row = 0; row < packed.shape[0]; ++row) {
        std::memmove(
            packed.bytes.data() + row * retained->components * item_bytes,
            packed.bytes.data()
                + (row * outputs_.width + retained->feature_offset) * item_bytes,
            retained->components * item_bytes);
    }
    packed.bytes.resize(checked_bytes(packed.shape[0], retained->components, item_bytes));
    packed.name = retained->name;
    packed.shape = {packed.shape.front()};
    if (retained->components != 1) {
        packed.shape.push_back(retained->components);
    }
    outputs.emplace(retained->name, std::move(packed));
    return outputs;
}

std::uint64_t Smedja::input_width() const noexcept { return inputs_.width; }
std::uint64_t Smedja::output_width() const noexcept { return outputs_.width; }
fjord::Element Smedja::input_element() const noexcept { return inputs_.element; }
fjord::Element Smedja::output_element() const noexcept { return outputs_.element; }

} // namespace foamnordic::inference
