#pragma once

#include <cstdint>

#include "bindnordic/tensor.hpp"

namespace bindnordic {

struct BackendCapabilities {
    bool concurrent_evaluation{false};
    bool row_independent{false};
    bool owns_compute_threads{false};
};

class PackedModelKernel {
public:
    virtual ~PackedModelKernel() = default;

    [[nodiscard]] virtual BackendCapabilities capabilities() const noexcept {
        return {};
    }

    [[nodiscard]] virtual Tensor evaluate(
        TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) = 0;
};

}  // namespace bindnordic
