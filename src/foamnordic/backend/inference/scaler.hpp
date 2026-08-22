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

#include <optional>
#include <vector>

#include "foamnordic/fjord/tensor.hpp"

namespace foamnordic::closure {

enum class ScalerKind {
    standard,
    minmax,
    robust,
};

class AffineScaler {
public:
    AffineScaler(
        ScalerKind kind,
        std::vector<double> gain,
        std::vector<double> bias,
        std::optional<double> clip_lower = std::nullopt,
        std::optional<double> clip_upper = std::nullopt);

    [[nodiscard]] static AffineScaler standard(
        std::vector<double> mean,
        std::vector<double> scale);
    [[nodiscard]] static AffineScaler minmax(
        std::vector<double> scale,
        std::vector<double> minimum,
        std::optional<double> feature_lower = std::nullopt,
        std::optional<double> feature_upper = std::nullopt);
    [[nodiscard]] static AffineScaler robust(
        std::vector<double> center,
        std::vector<double> scale);

    void transform(fjord::MutableTensorView values) const;
    void inverse_transform(fjord::MutableTensorView values) const;

    [[nodiscard]] ScalerKind kind() const noexcept;
    [[nodiscard]] std::size_t features() const noexcept;
    [[nodiscard]] const std::vector<double>& gain() const noexcept;
    [[nodiscard]] const std::vector<double>& bias() const noexcept;
    [[nodiscard]] std::optional<double> clip_lower() const noexcept;
    [[nodiscard]] std::optional<double> clip_upper() const noexcept;

private:
    void validate() const;
    void apply(fjord::MutableTensorView values, bool inverse) const;

    ScalerKind kind_;
    std::vector<double> gain_;
    std::vector<double> bias_;
    std::optional<double> clip_lower_;
    std::optional<double> clip_upper_;
};

}  // namespace foamnordic::closure
