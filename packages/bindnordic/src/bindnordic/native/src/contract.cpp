#include "bindnordic/contract.hpp"

#include <stdexcept>
#include <unordered_set>

namespace bindnordic {

void FieldContract::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("Model field name must not be empty.");
    }
    if (element_size(element) == 0) {
        throw std::invalid_argument("Model field element type is unsupported.");
    }
    if (components == 0) {
        throw std::invalid_argument("Model field components must be positive.");
    }
}

void ProgramContract::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("Model contract name must not be empty.");
    }
    if (inputs.empty() || outputs.empty()) {
        throw std::invalid_argument("Model contract requires inputs and outputs.");
    }
    for (const auto* fields : {&inputs, &outputs}) {
        std::unordered_set<std::string> names;
        for (const auto& field : *fields) {
            field.validate();
            if (!names.insert(field.name).second) {
                throw std::invalid_argument(
                    "Model field names must be unique within each direction.");
            }
        }
    }
}

}  // namespace bindnordic
