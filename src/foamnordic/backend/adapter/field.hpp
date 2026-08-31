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
#include <limits>
#include <vector>

#include "foamnordic/fjord/tensor.hpp"

namespace foamnordic::adapter {

struct FieldTransform {
    std::vector<double> scale{1.0};
    std::vector<double> bias{0.0};
    double lower{-std::numeric_limits<double>::infinity()};
    double upper{std::numeric_limits<double>::infinity()};

    void validate(std::uint64_t components) const;
    [[nodiscard]] bool is_identity(std::uint64_t components) const;
    void apply(fjord::TensorView source, fjord::MutableTensorView destination) const;
    void apply_in_place(fjord::MutableTensorView field) const;
};

struct FieldStatistics {
    double minimum{0.0};
    double maximum{0.0};
    double mean{0.0};
    std::uint64_t count{0};
    double l2{0.0};
};

[[nodiscard]] std::uint64_t field_components(const fjord::TensorView& field);
[[nodiscard]] FieldStatistics statistics(const fjord::TensorView& field);

}  // namespace foamnordic::adapter
