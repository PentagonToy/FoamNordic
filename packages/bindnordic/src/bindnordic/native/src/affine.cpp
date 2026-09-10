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

#include "bindnordic/affine.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bindnordic {
namespace {

template<class Value>
Tensor evaluate_values(
    TensorView features,
    std::uint64_t output_features,
    const std::vector<double>& weights,
    const std::vector<double>& bias) {
    const auto rows = features.shape.front();
    const auto input_features = features.shape[1];
    if (rows > std::numeric_limits<std::size_t>::max() / output_features
        || rows * output_features
               > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
        throw std::overflow_error("Dense affine output allocation overflowed.");
    }
    std::vector<std::byte> output(
        static_cast<std::size_t>(rows * output_features) * sizeof(Value));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t target = 0; target < output_features; ++target) {
            double result = bias[target];
            const auto weight_offset = target * input_features;
            for (std::uint64_t feature = 0; feature < input_features; ++feature) {
                Value source{};
                const auto source_offset =
                    static_cast<std::size_t>(row * input_features + feature)
                    * sizeof(Value);
                std::memcpy(
                    &source,
                    features.bytes.data() + source_offset,
                    sizeof(Value));
                result += static_cast<double>(source)
                          * weights[weight_offset + feature];
            }
            const auto converted = static_cast<Value>(result);
            const auto destination_offset =
                static_cast<std::size_t>(row * output_features + target)
                * sizeof(Value);
            std::memcpy(
                output.data() + destination_offset,
                &converted,
                sizeof(Value));
        }
    }
    return {
        "bindnordic.predictions",
        features.element,
        {rows, output_features},
        std::move(output),
    };
}

}  // namespace

DenseAffineKernel::DenseAffineKernel(
    std::uint64_t input_features,
    std::uint64_t output_features,
    std::vector<double> weights,
    std::vector<double> bias)
    : input_features_(input_features),
      output_features_(output_features),
      weights_(std::move(weights)),
      bias_(std::move(bias)) {
    if (input_features_ == 0 || output_features_ == 0
        || input_features_ > std::numeric_limits<std::size_t>::max() / output_features_
        || weights_.size() != input_features_ * output_features_
        || bias_.size() != output_features_) {
        throw std::invalid_argument("Dense affine kernel dimensions are invalid.");
    }
    for (const auto value : weights_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Dense affine weights must be finite.");
        }
    }
    for (const auto value : bias_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Dense affine bias must be finite.");
        }
    }
}

BackendCapabilities DenseAffineKernel::capabilities() const noexcept {
    return {
        .concurrent_evaluation = true,
        .row_independent = true,
        .owns_compute_threads = false,
    };
}

Tensor DenseAffineKernel::evaluate(
    TensorView features,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t /*rank*/) {
    features.validate();
    if (features.shape.size() != 2 || features.shape[1] != input_features_) {
        throw std::invalid_argument("Dense affine input does not match its contract.");
    }
    switch (features.element) {
        case Element::float32:
            return evaluate_values<float>(
                features,
                output_features_,
                weights_,
                bias_);
        case Element::float64:
            return evaluate_values<double>(
                features,
                output_features_,
                weights_,
                bias_);
        case Element::int32:
        case Element::int64:
            throw std::invalid_argument(
                "Dense affine kernel requires floating-point features.");
    }
    throw std::invalid_argument("Dense affine feature type is unsupported.");
}

std::uint64_t DenseAffineKernel::input_features() const noexcept {
    return input_features_;
}

std::uint64_t DenseAffineKernel::output_features() const noexcept {
    return output_features_;
}

}  // namespace bindnordic
