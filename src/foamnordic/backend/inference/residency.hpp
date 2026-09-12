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

#include "bindnordic/kernel.hpp"

namespace foamnordic::inference {

enum class ResidencyRoute : std::uint8_t {
    direct = 0,
    staging = 1,
    unsupported = 2,
};

struct ResidencySupport {
    bool pinned_host_staging{false};
    bool device_staging{false};
    bool unified_memory{false};
    bool event_completion{false};
};

struct ResidencyPlan {
    ResidencyRoute input{ResidencyRoute::unsupported};
    ResidencyRoute output{ResidencyRoute::unsupported};
    bool completion_supported{false};
    bool wait_for_event{false};
    std::uint64_t preferred_batch_rows{0};
    std::uint64_t alignment_bytes{1};

    [[nodiscard]] bool supported() const noexcept;
    [[nodiscard]] bool direct() const noexcept;
};

[[nodiscard]] ResidencyPlan plan_residency(
    const bindnordic::BackendCapabilities& capabilities,
    ResidencySupport support = {}) noexcept;

}  // namespace foamnordic::inference
