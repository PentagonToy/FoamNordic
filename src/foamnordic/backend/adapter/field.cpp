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

#include "foamnordic/backend/adapter/field.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace foamnordic::adapter {
namespace {

double component_value(const std::vector<double>& values, std::uint64_t component) {
    return values.size() == 1 ? values.front() : values[component];
}

template <class Value>
void transform_values(
    fjord::TensorView source,
    fjord::MutableTensorView destination,
    const FieldTransform& transform,
    std::uint64_t components) {
    const auto count = source.bytes.size() / sizeof(Value);
    for (std::size_t index = 0; index < count; ++index) {
        Value value{};
        std::memcpy(&value, source.bytes.data() + index * sizeof(Value), sizeof(Value));
        const auto component = static_cast<std::uint64_t>(index) % components;
        double result = static_cast<double>(value) * component_value(transform.scale, component)
                        + component_value(transform.bias, component);
        result = std::clamp(result, transform.lower, transform.upper);
        const auto converted = static_cast<Value>(result);
        std::memcpy(
            destination.bytes.data() + index * sizeof(Value), &converted, sizeof(Value));
    }
}

template <class Value>
FieldStatistics reduce_values(fjord::TensorView field) {
    const auto count = field.bytes.size() / sizeof(Value);
    if (count == 0) {
        throw std::invalid_argument("Cannot reduce an empty FoamNordic field.");
    }
    FieldStatistics result{
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        0.0,
        static_cast<std::uint64_t>(count),
    };
    long double sum = 0.0;
    long double square_sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        Value value{};
        std::memcpy(&value, field.bytes.data() + index * sizeof(Value), sizeof(Value));
        const auto converted = static_cast<double>(value);
        result.minimum = std::min(result.minimum, converted);
        result.maximum = std::max(result.maximum, converted);
        sum += static_cast<long double>(converted);
        square_sum += static_cast<long double>(converted)
                      * static_cast<long double>(converted);
    }
    result.mean = static_cast<double>(sum / static_cast<long double>(count));
    result.l2 = std::sqrt(static_cast<double>(square_sum));
    return result;
}

}  // namespace

std::uint64_t field_components(const fjord::TensorView& field) {
    field.validate();
    std::uint64_t components = 1;
    for (std::size_t index = 1; index < field.shape.size(); ++index) {
        if (field.shape[index] != 0
            && components > std::numeric_limits<std::uint64_t>::max() / field.shape[index]) {
            throw std::overflow_error("FoamNordic field component count overflowed.");
        }
        components *= field.shape[index];
    }
    return components;
}

void FieldTransform::validate(std::uint64_t components) const {
    if (components == 0 || (scale.size() != 1 && scale.size() != components)
        || (bias.size() != 1 && bias.size() != components) || scale.empty() || bias.empty()
        || std::isnan(lower) || std::isnan(upper) || lower > upper) {
        throw std::invalid_argument("FoamNordic field transform configuration is invalid.");
    }
}

bool FieldTransform::is_identity(std::uint64_t components) const {
    validate(components);
    return std::all_of(scale.begin(), scale.end(), [](double value) { return value == 1.0; })
           && std::all_of(bias.begin(), bias.end(), [](double value) { return value == 0.0; })
           && std::isinf(lower) && lower < 0.0 && std::isinf(upper) && upper > 0.0;
}

void FieldTransform::apply(
    fjord::TensorView source,
    fjord::MutableTensorView destination) const {
    source.validate();
    destination.validate();
    if (source.element != destination.element || source.shape != destination.shape
        || source.bytes.size() != destination.bytes.size()) {
        throw std::invalid_argument("FoamNordic transform source and destination differ.");
    }
    const auto components = field_components(source);
    validate(components);
    if (is_identity(components)) {
        if (source.bytes.data() != destination.bytes.data()) {
            std::memcpy(destination.bytes.data(), source.bytes.data(), source.bytes.size());
        }
        return;
    }
    switch (source.element) {
        case fjord::Element::float32:
            transform_values<float>(source, destination, *this, components);
            return;
        case fjord::Element::float64:
            transform_values<double>(source, destination, *this, components);
            return;
        case fjord::Element::int32:
        case fjord::Element::int64:
            throw std::invalid_argument("Affine field transforms require floating-point data.");
    }
}

void FieldTransform::apply_in_place(fjord::MutableTensorView field) const {
    apply(field.read_only(), field);
}

FieldStatistics statistics(const fjord::TensorView& field) {
    field.validate();
    switch (field.element) {
        case fjord::Element::float32:
            return reduce_values<float>(field);
        case fjord::Element::float64:
            return reduce_values<double>(field);
        case fjord::Element::int32:
            return reduce_values<std::int32_t>(field);
        case fjord::Element::int64:
            return reduce_values<std::int64_t>(field);
    }
    throw std::invalid_argument("Unsupported FoamNordic field element type.");
}

}  // namespace foamnordic::adapter
