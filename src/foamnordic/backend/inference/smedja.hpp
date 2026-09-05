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
#include <string>
#include <vector>

#include "foamnordic/backend/inference/exchange.hpp"

namespace foamnordic::inference {

class SmedjaWorkspace final {
public:
    [[nodiscard]] std::size_t retained_bytes() const noexcept;

private:
    friend class Smedja;
    fjord::Tensor input_;
};

// Smedja owns only immutable model-layout metadata. Tensor addresses are
// resolved afresh for every invocation so dynamic OpenFOAM storage is never
// retained beyond the request that supplied it.
class Smedja final {
  public:
    explicit Smedja(const ProgramContract& contract);

    [[nodiscard]] fjord::Tensor pack(const TensorMap& inputs,
                                     const std::vector<std::uint64_t>& active_cells,
                                     std::uint64_t exchange_index, double physical_time) const;

    [[nodiscard]] fjord::Tensor& pack_into(
        SmedjaWorkspace& workspace,
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time) const;

    [[nodiscard]] TensorMap unpack(fjord::Tensor packed) const;

    [[nodiscard]] std::uint64_t input_width() const noexcept;
    [[nodiscard]] std::uint64_t output_width() const noexcept;
    [[nodiscard]] fjord::Element input_element() const noexcept;
    [[nodiscard]] fjord::Element output_element() const noexcept;

  private:
    struct FieldLayout {
        std::string name;
        fjord::Element element{fjord::Element::float64};
        std::uint64_t components{1};
        std::uint64_t feature_offset{0};
    };

    struct Layout {
        fjord::Element element{fjord::Element::float64};
        std::uint64_t width{0};
        std::vector<FieldLayout> fields;
    };

    [[nodiscard]] static Layout compile(const std::vector<FieldContract>& fields);

    Layout inputs_;
    Layout outputs_;
};

} // namespace foamnordic::inference
