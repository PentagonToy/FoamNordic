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

#include "foamnordic/backend/inference/scaler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace foamnordic::closure {
namespace {

template <class Value>
void apply_values(
    fjord::MutableTensorView values,
    const std::vector<double>& gain,
    const std::vector<double>& bias,
    std::optional<double> clip_lower,
    std::optional<double> clip_upper,
    bool inverse) {
    const auto count = values.bytes.size() / sizeof(Value);
    for (std::size_t index = 0; index < count; ++index) {
        Value source{};
        std::memcpy(&source, values.bytes.data() + index * sizeof(Value), sizeof(Value));
        const auto feature = index % gain.size();
        double result = inverse
                            ? (static_cast<double>(source) - bias[feature]) / gain[feature]
                            : static_cast<double>(source) * gain[feature] + bias[feature];
        if (!inverse && clip_lower && clip_upper) {
            result = std::clamp(result, *clip_lower, *clip_upper);
        }
        const auto converted = static_cast<Value>(result);
        std::memcpy(
            values.bytes.data() + index * sizeof(Value), &converted, sizeof(Value));
    }
}

std::vector<double> reciprocal(const std::vector<double>& values) {
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.push_back(1.0 / value);
    }
    return result;
}

std::vector<double> scaled_bias(
    const std::vector<double>& center,
    const std::vector<double>& gain) {
    if (center.size() != gain.size()) {
        throw std::invalid_argument("Scaler center and scale feature counts differ.");
    }
    std::vector<double> result(center.size());
    for (std::size_t index = 0; index < center.size(); ++index) {
        result[index] = -center[index] * gain[index];
    }
    return result;
}

}  // namespace

AffineScaler::AffineScaler(
    ScalerKind kind,
    std::vector<double> gain,
    std::vector<double> bias,
    std::optional<double> clip_lower,
    std::optional<double> clip_upper)
    : kind_(kind),
      gain_(std::move(gain)),
      bias_(std::move(bias)),
      clip_lower_(clip_lower),
      clip_upper_(clip_upper) {
    validate();
}

AffineScaler AffineScaler::standard(
    std::vector<double> mean,
    std::vector<double> scale) {
    auto gain = reciprocal(scale);
    return {ScalerKind::standard, gain, scaled_bias(mean, gain)};
}

AffineScaler AffineScaler::minmax(
    std::vector<double> scale,
    std::vector<double> minimum,
    std::optional<double> feature_lower,
    std::optional<double> feature_upper) {
    return {
        ScalerKind::minmax,
        std::move(scale),
        std::move(minimum),
        feature_lower,
        feature_upper,
    };
}

AffineScaler AffineScaler::robust(
    std::vector<double> center,
    std::vector<double> scale) {
    auto gain = reciprocal(scale);
    return {ScalerKind::robust, gain, scaled_bias(center, gain)};
}

void AffineScaler::validate() const {
    if (gain_.empty() || gain_.size() != bias_.size()) {
        throw std::invalid_argument("Scaler gain and bias feature counts are invalid.");
    }
    for (std::size_t index = 0; index < gain_.size(); ++index) {
        if (!std::isfinite(gain_[index]) || gain_[index] == 0.0
            || !std::isfinite(bias_[index])) {
            throw std::invalid_argument("Scaler affine coefficients are invalid.");
        }
    }
    if (clip_lower_.has_value() != clip_upper_.has_value()
        || (clip_lower_ && (!std::isfinite(*clip_lower_) || !std::isfinite(*clip_upper_)
                            || *clip_lower_ > *clip_upper_))) {
        throw std::invalid_argument("Scaler clipping range is invalid.");
    }
}

void AffineScaler::apply(fjord::MutableTensorView values, bool inverse) const {
    values.validate();
    std::uint64_t features = 1;
    for (std::size_t index = 1; index < values.shape.size(); ++index) {
        features *= values.shape[index];
    }
    if (features != gain_.size()) {
        throw std::invalid_argument("Tensor feature count does not match its scaler.");
    }
    switch (values.element) {
        case fjord::Element::float32:
            apply_values<float>(
                values, gain_, bias_, clip_lower_, clip_upper_, inverse);
            return;
        case fjord::Element::float64:
            apply_values<double>(
                values, gain_, bias_, clip_lower_, clip_upper_, inverse);
            return;
        case fjord::Element::int32:
        case fjord::Element::int64:
            throw std::invalid_argument("Native scaler requires floating-point tensors.");
    }
}

void AffineScaler::transform(fjord::MutableTensorView values) const {
    apply(values, false);
}

void AffineScaler::inverse_transform(fjord::MutableTensorView values) const {
    apply(values, true);
}

ScalerKind AffineScaler::kind() const noexcept { return kind_; }
std::size_t AffineScaler::features() const noexcept { return gain_.size(); }
const std::vector<double>& AffineScaler::gain() const noexcept { return gain_; }
const std::vector<double>& AffineScaler::bias() const noexcept { return bias_; }
std::optional<double> AffineScaler::clip_lower() const noexcept { return clip_lower_; }
std::optional<double> AffineScaler::clip_upper() const noexcept { return clip_upper_; }

}  // namespace foamnordic::closure
