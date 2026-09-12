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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bindnordic/tensor.hpp"

namespace bindnordic {

struct FieldContract {
    std::string name;
    Element element{Element::float64};
    std::uint64_t components{1};

    void validate() const;
};

struct ProgramContract {
    std::string name;
    std::vector<FieldContract> inputs;
    std::vector<FieldContract> outputs;

    void validate() const;
};

}  // namespace bindnordic
