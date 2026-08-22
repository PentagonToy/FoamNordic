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
#include <optional>
#include <string>
#include <vector>

#include "foamnordic/backend/inference/closure.hpp"
#include "foamnordic/backend/inference/scaler.hpp"

namespace foamnordic::closure {

enum class ModelFormat {
    equinox,
    joblib,
    onnx,
};

struct TreeLeaf {
    std::string path;
    fjord::Element element{fjord::Element::float64};
    std::vector<std::uint64_t> shape;
    std::uint64_t byte_offset{0};
    std::uint64_t byte_count{0};

    void validate() const;
};

struct ModelArtifact {
    std::uint32_t schema_version{1};
    ModelFormat format{ModelFormat::onnx};
    std::string artifact_path;
    ClosureContract contract;
    std::vector<TreeLeaf> tree_leaves;
    std::optional<AffineScaler> input_scaler;
    std::optional<AffineScaler> output_scaler;

    void validate() const;
};

[[nodiscard]] const char* name(ModelFormat format) noexcept;

}  // namespace foamnordic::closure
