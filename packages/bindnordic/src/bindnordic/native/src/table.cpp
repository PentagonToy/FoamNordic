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
    const std::vector<TableAxis>& axes,
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
        stride *= axes[axis].count;
    }

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::size_t axis = 0; axis < dimensions; ++axis) {
            auto coordinate = input_value<Value>(features, row * dimensions + axis);
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument("Table coordinate must be finite.");
            }
            const auto& descriptor = axes[axis];
            const auto first = descriptor.uniform
                ? descriptor.start : descriptor.points.front();
            const auto last = descriptor.uniform
                ? descriptor.start + descriptor.step * (descriptor.count - 1)
                : descriptor.points.back();
            if (coordinate < first || coordinate > last) {
                if (!clamp[axis]) {
                    throw std::out_of_range("Table coordinate is outside its axis.");
                }
                coordinate = std::clamp(coordinate, first, last);
            }
            std::size_t lo = 0;
            if (descriptor.uniform) {
                lo = std::min(
                    descriptor.count - 2,
                    static_cast<std::size_t>(
                        std::max(0.0, std::floor(
                            (coordinate - descriptor.start) / descriptor.step))));
            } else {
                const auto upper = std::upper_bound(
                    descriptor.points.begin(), descriptor.points.end(), coordinate);
                lo = upper == descriptor.points.begin() ? 0
                    : upper == descriptor.points.end() ? descriptor.count - 2
                    : static_cast<std::size_t>(
                          upper - descriptor.points.begin() - 1);
            }
            const auto lower_point = descriptor.uniform
                ? descriptor.start + descriptor.step * lo
                : descriptor.points[lo];
            const auto upper_point = descriptor.uniform
                ? lower_point + descriptor.step
                : descriptor.points[lo + 1];
            lower[axis] = lo;
            fraction[axis] =
                (coordinate - lower_point) / (upper_point - lower_point);
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
    std::array<char, 8> magic{};
    for (auto& value : magic) value = reader.take<char>();
    const auto revision = magic == std::array<char, 8>{'F', 'N', 'O', 'T', 'A', 'B', '1', '\0'} ? 1
        : magic == std::array<char, 8>{'F', 'N', 'O', 'T', 'A', 'B', '2', '\0'} ? 2
        : 0;
    if (revision == 0) {
        throw std::invalid_argument("Table payload magic is invalid.");
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
        const auto kind = revision == 1 ? std::uint8_t{0} : reader.take<std::uint8_t>();
        const auto axis_reserved = revision == 1 ? std::uint16_t{0}
                                                  : reader.take<std::uint16_t>();
        const auto count = reader.take<std::uint64_t>();
        if (policy > 1 || kind > 1 || axis_reserved != 0 || count < 2
            || entries > std::numeric_limits<std::size_t>::max() / count) {
            throw std::invalid_argument("Table axis metadata is invalid.");
        }
        entries *= count;
        clamp_.push_back(policy == 0);
        auto& descriptor = axes_.emplace_back();
        descriptor.count = count;
        descriptor.uniform = kind == 1;
        if (descriptor.uniform) {
            descriptor.start = reader.take<double>();
            descriptor.step = reader.take<double>();
            const auto last = descriptor.start + descriptor.step * (count - 1);
            const auto penultimate =
                descriptor.start + descriptor.step * (count - 2);
            const auto interval_end = penultimate + descriptor.step;
            if (!std::isfinite(descriptor.start) || !std::isfinite(descriptor.step)
                || descriptor.step <= 0.0 || !std::isfinite(last)
                || !std::isfinite(penultimate) || !std::isfinite(interval_end)
                || interval_end <= penultimate) {
                throw std::invalid_argument("Uniform table axis is invalid.");
            }
            continue;
        }
        descriptor.points.reserve(count);
        for (std::uint64_t index = 0; index < count; ++index) {
            const auto point = reader.take<double>();
            if (!std::isfinite(point)
                || (!descriptor.points.empty()
                    && point <= descriptor.points.back())) {
                throw std::invalid_argument("Table axes must be finite and strictly increasing.");
            }
            descriptor.points.push_back(point);
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
