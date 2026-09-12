// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * FoamNordic
 */
// clang-format on

#pragma once

#include "bindnordic/tensor.hpp"

namespace foamnordic::fjord {

using bindnordic::Element;
using bindnordic::as_bytes;
using bindnordic::as_writable_bytes;
using bindnordic::element_size;

struct TensorView {
    std::string name;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::span<const std::byte> bytes;
    std::uint64_t time_index{0};
    double physical_time{0.0};
    std::uint64_t solver_time_index{0};
    std::uint64_t topology_generation{0};

    [[nodiscard]] bindnordic::TensorView model_view() const {
        return {name, element, shape, bytes};
    }

    [[nodiscard]] operator bindnordic::TensorView() const {
        return model_view();
    }

    [[nodiscard]] std::uint64_t element_count() const {
        return model_view().element_count();
    }

    void validate() const { model_view().validate(); }
};

struct Tensor {
    std::string name;
    Element element{Element::float64};
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes;
    std::uint64_t time_index{0};
    double physical_time{0.0};
    std::uint64_t solver_time_index{0};
    std::uint64_t topology_generation{0};

    [[nodiscard]] TensorView view() const {
        return {
            name,
            element,
            shape,
            bytes,
            time_index,
            physical_time,
            solver_time_index,
            topology_generation};
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
    std::uint64_t topology_generation{0};

    [[nodiscard]] TensorView read_only() const {
        return {
            name,
            element,
            shape,
            bytes,
            time_index,
            physical_time,
            solver_time_index,
            topology_generation};
    }

    [[nodiscard]] bindnordic::MutableTensorView model_view() const {
        return {name, element, shape, bytes};
    }

    [[nodiscard]] operator bindnordic::MutableTensorView() const {
        return model_view();
    }

    void validate() const { read_only().validate(); }
};

}  // namespace foamnordic::fjord
