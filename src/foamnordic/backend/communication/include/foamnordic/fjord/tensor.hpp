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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace foamnordic::fjord {

enum class Element : std::uint8_t {
    float32 = 1,
    float64 = 2,
    int32 = 3,
    int64 = 4,
};

[[nodiscard]] constexpr std::size_t element_size(Element element) {
    switch (element) {
        case Element::float32:
        case Element::int32:
            return 4;
        case Element::float64:
        case Element::int64:
            return 8;
    }
    return 0;
}

struct TensorView {
    std::string name;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::span<const std::byte> bytes;
    std::uint64_t time_index{0};
    double physical_time{0.0};
    std::uint64_t solver_time_index{0};

    [[nodiscard]] std::uint64_t element_count() const {
        std::uint64_t count = 1;
        for (const auto extent : shape) {
            if (extent == 0) {
                return 0;
            }
            if (count > std::numeric_limits<std::uint64_t>::max() / extent) {
                throw std::overflow_error("Tensor shape exceeds the protocol limit.");
            }
            count *= extent;
        }
        return count;
    }

    void validate() const {
        if (name.empty()) {
            throw std::invalid_argument("Tensor name must not be empty.");
        }
        if (shape.empty()) {
            throw std::invalid_argument("Tensor shape must not be empty.");
        }
        const auto expected = element_count() * element_size(element);
        if (expected != bytes.size()) {
            throw std::invalid_argument("Tensor byte count does not match its shape and element type.");
        }
    }
};

struct Tensor {
    std::string name;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes;
    std::uint64_t time_index{0};
    double physical_time{0.0};
    std::uint64_t solver_time_index{0};

    [[nodiscard]] TensorView view() const {
        return TensorView{
            name,
            element,
            shape,
            bytes,
            time_index,
            physical_time,
            solver_time_index};
    }
};

struct MutableTensorView {
    std::string name;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::span<std::byte> bytes;
    std::uint64_t time_index{0};
    double physical_time{0.0};
    std::uint64_t solver_time_index{0};

    [[nodiscard]] TensorView read_only() const {
        return TensorView{
            name,
            element,
            shape,
            bytes,
            time_index,
            physical_time,
            solver_time_index};
    }

    void validate() const { read_only().validate(); }
};

template <class Value, std::size_t Extent>
[[nodiscard]] std::span<const std::byte> as_bytes(std::span<Value, Extent> values) {
    static_assert(std::is_trivially_copyable_v<std::remove_cv_t<Value>>);
    const auto bytes = std::as_bytes(values);
    return {bytes.data(), bytes.size()};
}

template <class Value, std::size_t Extent>
[[nodiscard]] std::span<std::byte> as_writable_bytes(std::span<Value, Extent> values) {
    static_assert(!std::is_const_v<Value>);
    static_assert(std::is_trivially_copyable_v<Value>);
    const auto bytes = std::as_writable_bytes(values);
    return {bytes.data(), bytes.size()};
}

}  // namespace foamnordic::fjord
