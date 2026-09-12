// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * BindNordic
 */
// clang-format on

#include "bindnordic/table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bindnordic {
namespace {

class Reader {
public:
    explicit Reader(std::span<const std::byte> data) : data_(data) {}

    template<class Value>
    Value take() {
        if (offset_ > data_.size() || sizeof(Value) > data_.size() - offset_) {
            throw std::invalid_argument("Table payload is truncated.");
        }
        Value value{};
        std::memcpy(&value, data_.data() + offset_, sizeof(Value));
        offset_ += sizeof(Value);
        return value;
    }

    [[nodiscard]] std::size_t remaining() const noexcept {
        return data_.size() - offset_;
    }

private:
    std::span<const std::byte> data_;
    std::size_t offset_{0};
};

template<class Value>
double input_value(TensorView input, std::uint64_t index) {
    Value value{};
    std::memcpy(&value, input.bytes.data() + index * sizeof(Value), sizeof(Value));
    return static_cast<double>(value);
}

template<class Value>
Tensor interpolate(
    TensorView features,
    const std::vector<std::vector<double>>& axes,
    const std::vector<bool>& clamp,
    const std::vector<double>& table,
    std::uint64_t output_features) {
    const auto rows = features.shape[0];
    const auto dimensions = axes.size();
    if (rows > std::numeric_limits<std::size_t>::max() / output_features) {
        throw std::overflow_error("Table output allocation overflowed.");
    }
    std::vector<std::byte> output(
        static_cast<std::size_t>(rows * output_features) * sizeof(Value));
    std::vector<std::size_t> lower(dimensions);
    std::vector<double> fraction(dimensions);
    std::vector<std::size_t> strides(dimensions);
    std::size_t stride = output_features;
    for (std::size_t axis = dimensions; axis-- > 0;) {
        strides[axis] = stride;
        stride *= axes[axis].size();
    }

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::size_t axis = 0; axis < dimensions; ++axis) {
            auto coordinate = input_value<Value>(features, row * dimensions + axis);
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument("Table coordinate must be finite.");
            }
            const auto& points = axes[axis];
            if (coordinate < points.front() || coordinate > points.back()) {
                if (!clamp[axis]) {
                    throw std::out_of_range("Table coordinate is outside its axis.");
                }
                coordinate = std::clamp(coordinate, points.front(), points.back());
            }
            const auto upper = std::upper_bound(points.begin(), points.end(), coordinate);
            const auto lo = upper == points.begin() ? 0
                : upper == points.end() ? points.size() - 2
                                        : static_cast<std::size_t>(upper - points.begin() - 1);
            lower[axis] = lo;
            fraction[axis] = (coordinate - points[lo]) / (points[lo + 1] - points[lo]);
        }

        const auto corners = std::size_t{1} << dimensions;
        for (std::uint64_t target = 0; target < output_features; ++target) {
            double result = 0.0;
            for (std::size_t corner = 0; corner < corners; ++corner) {
                double weight = 1.0;
                std::size_t index = target;
                for (std::size_t axis = 0; axis < dimensions; ++axis) {
                    const auto high = (corner >> axis) & 1U;
                    weight *= high ? fraction[axis] : 1.0 - fraction[axis];
                    index += (lower[axis] + high) * strides[axis];
                }
                result += weight * table[index];
            }
            const auto converted = static_cast<Value>(result);
            std::memcpy(
                output.data() + static_cast<std::size_t>(row * output_features + target) * sizeof(Value),
                &converted,
                sizeof(Value));
        }
    }
    return {"bindnordic.predictions", features.element, {rows, output_features}, std::move(output)};
}

}  // namespace

TableKernel::TableKernel(
    std::span<const std::byte> payload,
    std::uint64_t input_features,
    std::uint64_t output_features,
    Element element)
    : element_(element), output_features_(output_features) {
    Reader reader(payload);
    constexpr std::array<char, 8> magic{'F', 'N', 'O', 'T', 'A', 'B', '1', '\0'};
    for (const auto expected : magic) {
        if (reader.take<char>() != expected) {
            throw std::invalid_argument("Table payload magic is invalid.");
        }
    }
    const auto dtype = reader.take<std::uint8_t>();
    const auto dimensions = reader.take<std::uint8_t>();
    const auto reserved = reader.take<std::uint16_t>();
    const auto outputs = reader.take<std::uint64_t>();
    if (reserved != 0 || dimensions == 0 || dimensions > 8
        || dimensions != input_features || outputs != output_features
        || dtype != static_cast<std::uint8_t>(element)) {
        throw std::invalid_argument("Table payload does not match its FNOM contract.");
    }
    std::size_t entries = output_features;
    axes_.reserve(dimensions);
    clamp_.reserve(dimensions);
    for (std::uint8_t axis = 0; axis < dimensions; ++axis) {
        const auto policy = reader.take<std::uint8_t>();
        const auto count = reader.take<std::uint64_t>();
        if (policy > 1 || count < 2
            || entries > std::numeric_limits<std::size_t>::max() / count) {
            throw std::invalid_argument("Table axis metadata is invalid.");
        }
        entries *= count;
        clamp_.push_back(policy == 0);
        auto& points = axes_.emplace_back();
        points.reserve(count);
        for (std::uint64_t index = 0; index < count; ++index) {
            const auto point = reader.take<double>();
            if (!std::isfinite(point) || (!points.empty() && point <= points.back())) {
                throw std::invalid_argument("Table axes must be finite and strictly increasing.");
            }
            points.push_back(point);
        }
    }
    const auto value_bytes = element_size(element_);
    if (reader.remaining() != entries * value_bytes) {
        throw std::invalid_argument("Table value storage does not match its axes.");
    }
    values_.reserve(entries);
    for (std::size_t index = 0; index < entries; ++index) {
        const auto value = element_ == Element::float32
                               ? static_cast<double>(reader.take<float>())
                               : reader.take<double>();
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Table values must be finite.");
        }
        values_.push_back(value);
    }
}

BackendCapabilities TableKernel::capabilities() const noexcept {
    return {.concurrent_evaluation = true, .row_independent = true, .owns_compute_threads = false};
}

Tensor TableKernel::evaluate(
    TensorView features,
    std::uint64_t,
    double,
    std::uint32_t) {
    features.validate();
    if (features.shape.size() != 2 || features.shape[1] != axes_.size()
        || features.element != element_) {
        throw std::invalid_argument("Table input does not match its FNOM contract.");
    }
    if (element_ == Element::float32) {
        return interpolate<float>(features, axes_, clamp_, values_, output_features_);
    }
    if (element_ == Element::float64) {
        return interpolate<double>(features, axes_, clamp_, values_, output_features_);
    }
    throw std::invalid_argument("Table kernels require floating-point input.");
}

}  // namespace bindnordic
