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
