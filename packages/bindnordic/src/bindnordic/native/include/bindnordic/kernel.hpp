#pragma once

#include <cstdint>

#include "bindnordic/tensor.hpp"

namespace bindnordic {

enum class MemorySpace : std::uint8_t {
    host = 0,
    pinned_host = 1,
    device = 2,
    unified = 3,
};

enum class CompletionMode : std::uint8_t {
    synchronous = 0,
    event = 1,
};

struct BackendCapabilities {
    bool concurrent_evaluation{false};
    bool row_independent{false};
    bool owns_compute_threads{false};
    MemorySpace input_memory{MemorySpace::host};
    MemorySpace output_memory{MemorySpace::host};
    bool persistent_device_state{false};
    CompletionMode completion{CompletionMode::synchronous};
    bool workspace_recreated_on_shape_change{false};
    std::uint64_t preferred_batch_rows{0};
    std::uint64_t alignment_bytes{1};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return alignment_bytes != 0
               && (alignment_bytes & (alignment_bytes - 1)) == 0;
    }
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
