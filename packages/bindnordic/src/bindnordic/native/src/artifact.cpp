// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * BindNordic
 */
// clang-format on

#include "bindnordic/artifact.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace bindnordic {

void TreeLeaf::validate() const {
    if (path.empty() || shape.empty() || byte_count == 0) {
        throw std::invalid_argument("Equinox tree leaf metadata is incomplete.");
    }
    std::uint64_t elements = 1;
    for (const auto extent : shape) {
        if (extent == 0 || elements > std::numeric_limits<std::uint64_t>::max() / extent) {
            throw std::invalid_argument("Equinox tree leaf shape is invalid.");
        }
        elements *= extent;
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_size(element)
        || elements * element_size(element) != byte_count) {
        throw std::invalid_argument("Equinox tree leaf byte count is invalid.");
    }
}

void ModelArtifact::validate() const {
    if (artifact_path.empty()) {
        throw std::invalid_argument("BindNordic model artifact metadata is invalid.");
    }
    if (format == ModelFormat::compiled && runtime != "compiled" && runtime != "forest") {
        throw std::invalid_argument("Compiled artifact runtime is invalid.");
    }
    if (!runtime.empty() && format != ModelFormat::compiled) {
        const auto valid_joblib = format == ModelFormat::joblib
                                  && (runtime == "sklearn" || runtime == "sklearnex");
        const auto valid_table = format == ModelFormat::table && runtime == "table";
        if (!valid_joblib && !valid_table) {
            throw std::invalid_argument("Model artifact runtime is invalid.");
        }
    }
    if (input_format != "numpy" && input_format != "pandas"
        && input_format != "polars") {
        throw std::invalid_argument("Model artifact input format is invalid.");
    }
    if (format != ModelFormat::joblib && input_format != "numpy") {
        throw std::invalid_argument("Only Joblib artifacts may use named input formats.");
    }
    if (input_format != "numpy"
        && std::ranges::any_of(contract.inputs, [](const auto& field) {
               return field.components != 1;
           })) {
        throw std::invalid_argument("Named input formats require scalar input ports.");
    }
    contract.validate();
    const auto feature_count = [](const std::vector<FieldContract>& fields) {
        std::uint64_t count = 0;
        for (const auto& field : fields) {
            if (count > std::numeric_limits<std::uint64_t>::max() - field.components) {
                throw std::overflow_error("Model feature count overflowed.");
            }
            count += field.components;
        }
        return count;
    };
    if (input_scaler && input_scaler->features() != feature_count(contract.inputs)) {
        throw std::invalid_argument("Input scaler does not match model input features.");
    }
    if (output_scaler && output_scaler->features() != feature_count(contract.outputs)) {
        throw std::invalid_argument("Output scaler does not match model output features.");
    }
    if (format != ModelFormat::equinox && !tree_leaves.empty()) {
        throw std::invalid_argument("Only Equinox artifacts may define tree leaves.");
    }
    if (format == ModelFormat::equinox && tree_leaves.empty()) {
        throw std::invalid_argument("Equinox artifact requires a flattened tree manifest.");
    }
    std::unordered_set<std::string> paths;
    std::uint64_t previous_end = 0;
    for (const auto& leaf : tree_leaves) {
        leaf.validate();
        if (!paths.insert(leaf.path).second || leaf.byte_offset < previous_end
            || leaf.byte_offset > std::numeric_limits<std::uint64_t>::max() - leaf.byte_count) {
            throw std::invalid_argument("Equinox tree leaves overlap or repeat.");
        }
        previous_end = leaf.byte_offset + leaf.byte_count;
    }
}

const char* name(ModelFormat format) noexcept {
    switch (format) {
        case ModelFormat::compiled:
            return "compiled";
        case ModelFormat::equinox:
            return "equinox";
        case ModelFormat::joblib:
            return "joblib";
        case ModelFormat::onnx:
            return "onnx";
        case ModelFormat::table:
            return "table";
    }
    return "unknown";
}

}  // namespace bindnordic
