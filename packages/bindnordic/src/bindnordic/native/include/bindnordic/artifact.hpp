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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bindnordic/contract.hpp"
#include "bindnordic/scaler.hpp"

namespace bindnordic {

enum class ModelFormat {
    compiled,
    equinox,
    joblib,
    onnx,
    table,
};

struct TreeLeaf {
    std::string path;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::uint64_t byte_offset{0};
    std::uint64_t byte_count{0};

    void validate() const;
};

struct ModelArtifact {
    ModelFormat format{ModelFormat::onnx};
    std::string artifact_path;
    ProgramContract contract;
    std::vector<TreeLeaf> tree_leaves;
    std::optional<AffineScaler> input_scaler;
    std::optional<AffineScaler> output_scaler;
    std::string runtime;

    void validate() const;
};

[[nodiscard]] const char* name(ModelFormat format) noexcept;

}  // namespace bindnordic
