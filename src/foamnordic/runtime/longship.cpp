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

#include "foamnordic/runtime/longship.hpp"

#include <limits>
#include <stdexcept>

namespace foamnordic::native {
namespace {

[[nodiscard]] std::uint32_t checked_add(
    std::uint32_t left,
    std::uint32_t right) {
    if (left > std::numeric_limits<std::uint32_t>::max() - right) {
        throw std::overflow_error("Longship CPU allocation overflowed.");
    }
    return left + right;
}

[[nodiscard]] std::uint32_t checked_multiply(
    std::uint32_t left,
    std::uint32_t right) {
    if (left != 0
        && right > std::numeric_limits<std::uint32_t>::max() / left) {
        throw std::overflow_error("Longship CPU allocation overflowed.");
    }
    return left * right;
}

}  // namespace

void LongshipRequest::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("Longship name must not be empty.");
    }
    if (solver_nodes == 0 || solver_tasks == 0) {
        throw std::invalid_argument(
            "Longship requires solver nodes and solver tasks.");
    }
    if (solver_tasks % solver_nodes != 0) {
        throw std::invalid_argument(
            "Longship solver tasks must divide evenly across solver nodes.");
    }
    if (solver_cpus_per_task == 0
        || (use_model_host && host_cpus_per_node == 0)) {
        throw std::invalid_argument("Longship CPU requests must be positive.");
    }
}

LongshipPlan plan_longship(const LongshipRequest& request) {
    request.validate();
    const auto tasks_per_node = request.solver_tasks / request.solver_nodes;
    const auto solver_cpus_per_node = checked_multiply(
        tasks_per_node, request.solver_cpus_per_task);
    if (!request.use_model_host) {
        return {
            request.name,
            {
                HostPlacement::attached,
                DataPath::unix_socket,
                0,
                true,
                true,
                false,
                "solver-only OpenFOAM workload",
            },
            request.solver_nodes,
            request.solver_tasks,
            tasks_per_node,
            request.solver_cpus_per_task,
            0,
            0,
            solver_cpus_per_node,
            false,
            false,
        };
    }
    auto placement_request = request.placement;
    placement_request.solver_nodes = request.solver_nodes;
    const auto placement = resolve_placement(placement_request);
    if (placement.placement != HostPlacement::attached) {
        throw std::invalid_argument(
            "Longship currently requires an attached ModelHost.");
    }
    if (!placement.same_allocation || !placement.same_node
        || placement.host_instances != request.solver_nodes) {
        throw std::logic_error(
            "Attached ModelHost placement violated the Longship contract.");
    }

    const auto allocation_cpus_per_node = checked_add(
        solver_cpus_per_node, request.host_cpus_per_node);

    return {
        request.name,
        placement,
        request.solver_nodes,
        request.solver_tasks,
        tasks_per_node,
        request.solver_cpus_per_task,
        placement.host_instances,
        request.host_cpus_per_node,
        allocation_cpus_per_node,
        true,
        true,
    };
}

}  // namespace foamnordic::native
