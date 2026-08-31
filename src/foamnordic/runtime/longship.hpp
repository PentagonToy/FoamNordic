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

#include "foamnordic/runtime/placement.hpp"

namespace foamnordic::native {

struct LongshipRequest {
    std::string name{"foamnordic"};
    std::uint32_t solver_nodes{1};
    std::uint32_t solver_tasks{1};
    std::uint32_t solver_cpus_per_task{1};
    std::uint32_t host_cpus_per_node{1};
    bool use_closure_host{true};
    PlacementRequest placement{};

    void validate() const;
};

struct LongshipPlan {
    std::string name;
    PlacementPlan placement;
    std::uint32_t allocation_nodes{1};
    std::uint32_t solver_tasks{1};
    std::uint32_t solver_tasks_per_node{1};
    std::uint32_t solver_cpus_per_task{1};
    std::uint32_t host_tasks{1};
    std::uint32_t host_cpus_per_task{1};
    std::uint32_t allocation_cpus_per_node{2};
    bool host_starts_first{true};
    bool fail_together{true};
};

[[nodiscard]] LongshipPlan plan_longship(const LongshipRequest& request);

}  // namespace foamnordic::native
