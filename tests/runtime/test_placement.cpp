#include <stdexcept>
#include <string>

#include "foamnordic/runtime/placement.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_attached_is_the_default() {
    const auto plan = foamnordic::native::resolve_placement({});
    require(
        plan.placement == foamnordic::native::HostPlacement::attached,
        "ClosureHost is not attached by default.");
    require(plan.host_instances == 1, "Single-node solver did not receive one ClosureHost.");
    require(plan.same_allocation && plan.same_node, "Attached ClosureHost was separated.");
    require(plan.coupled_lifetime, "Attached ClosureHost lifetime is not tied to the solver.");
    require(
        plan.data_path == foamnordic::native::DataPath::shared_memory,
        "Attached ClosureHost did not prefer SHM.");
}

void test_attached_scales_per_solver_node() {
    foamnordic::native::PlacementRequest request;
    request.solver_nodes = 4;
    request.shared_memory_available = false;
    const auto plan = foamnordic::native::resolve_placement(request);
    require(plan.host_instances == 4, "Multi-node solver lacks one ClosureHost per node.");
    require(
        plan.data_path == foamnordic::native::DataPath::unix_socket,
        "Attached placement did not fall back from SHM to UDS.");
}

void test_attached_portable_fallback() {
    foamnordic::native::PlacementRequest request;
    request.shared_memory_available = false;
    request.unix_socket_available = false;
    const auto plan = foamnordic::native::resolve_placement(request);
    require(
        plan.data_path == foamnordic::native::DataPath::tcp,
        "Attached placement lacks a portable fallback.");
}

void test_gpu_is_central_when_solver_nodes_have_no_gpu() {
    foamnordic::native::PlacementRequest request;
    request.device = foamnordic::native::InferenceDevice::gpu;
    request.solver_nodes_have_device = false;
    request.solver_nodes = 8;
    request.central_host_nodes = 2;
    request.ucx_available = true;
    const auto plan = foamnordic::native::resolve_placement(request);
    require(
        plan.placement == foamnordic::native::HostPlacement::central,
        "GPU host was not centralized.");
    require(plan.host_instances == 2, "Central ClosureHost node count is incorrect.");
    require(!plan.same_allocation && !plan.same_node, "Central ClosureHost is colocated.");
    require(plan.coupled_lifetime, "Central ClosureHost lost experiment lifetime coupling.");
    require(
        plan.data_path == foamnordic::native::DataPath::ucx,
        "Central ClosureHost did not prefer UCX.");
}

void test_gpu_attaches_when_solver_node_has_gpu() {
    foamnordic::native::PlacementRequest request;
    request.device = foamnordic::native::InferenceDevice::gpu;
    request.solver_nodes_have_device = true;
    const auto plan = foamnordic::native::resolve_placement(request);
    require(
        plan.placement == foamnordic::native::HostPlacement::attached,
        "GPU ClosureHost was centralized despite a local GPU.");
}

void test_central_rejects_local_only_paths() {
    foamnordic::native::PlacementRequest request;
    request.placement = foamnordic::native::HostPlacement::central;
    request.data_path = foamnordic::native::DataPathPreference::shared_memory;
    bool rejected = false;
    try {
        static_cast<void>(foamnordic::native::resolve_placement(request));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Central ClosureHost accepted SHM.");
}

}  // namespace

int main() {
    test_attached_is_the_default();
    test_attached_scales_per_solver_node();
    test_attached_portable_fallback();
    test_gpu_is_central_when_solver_nodes_have_no_gpu();
    test_gpu_attaches_when_solver_node_has_gpu();
    test_central_rejects_local_only_paths();
}
