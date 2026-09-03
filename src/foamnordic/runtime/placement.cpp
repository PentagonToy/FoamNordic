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

#include "foamnordic/runtime/placement.hpp"

#include <stdexcept>

namespace foamnordic::native {
namespace {

DataPath resolve_attached_path(const PlacementRequest& request) {
    switch (request.data_path) {
        case DataPathPreference::automatic:
            if (request.shared_memory_available) {
                return DataPath::shared_memory;
            }
            if (request.unix_socket_available) {
                return DataPath::unix_socket;
            }
            return DataPath::tcp;
        case DataPathPreference::shared_memory:
            if (!request.shared_memory_available) {
                throw std::runtime_error("Attached ModelHost requested unavailable SHM.");
            }
            return DataPath::shared_memory;
        case DataPathPreference::unix_socket:
            if (!request.unix_socket_available) {
                throw std::runtime_error("Attached ModelHost requested unavailable UDS.");
            }
            return DataPath::unix_socket;
        case DataPathPreference::ucx:
            if (!request.ucx_available) {
                throw std::runtime_error("ModelHost requested unavailable UCX.");
            }
            return DataPath::ucx;
        case DataPathPreference::tcp:
            return DataPath::tcp;
    }
    throw std::logic_error("Unknown FoamNordic data-path preference.");
}

DataPath resolve_central_path(const PlacementRequest& request) {
    switch (request.data_path) {
        case DataPathPreference::automatic:
            return request.ucx_available ? DataPath::ucx : DataPath::tcp;
        case DataPathPreference::ucx:
            if (!request.ucx_available) {
                throw std::runtime_error("Central ModelHost requested unavailable UCX.");
            }
            return DataPath::ucx;
        case DataPathPreference::tcp:
            return DataPath::tcp;
        case DataPathPreference::shared_memory:
        case DataPathPreference::unix_socket:
            throw std::invalid_argument(
                "Central ModelHost cannot use a same-node data path.");
    }
    throw std::logic_error("Unknown FoamNordic data-path preference.");
}

}  // namespace

void PlacementRequest::validate() const {
    if (solver_nodes == 0) {
        throw std::invalid_argument("FoamNordic requires at least one solver node.");
    }
    if (central_host_nodes == 0) {
        throw std::invalid_argument("Central ModelHost requires at least one host node.");
    }
}

PlacementPlan resolve_placement(const PlacementRequest& request) {
    request.validate();
    auto placement = request.placement;
    if (placement == HostPlacement::automatic) {
        placement = request.device == InferenceDevice::gpu
                            && !request.solver_nodes_have_device
                        ? HostPlacement::central
                        : HostPlacement::attached;
    }
    if (placement == HostPlacement::attached) {
        if (request.device == InferenceDevice::gpu && !request.solver_nodes_have_device) {
            throw std::invalid_argument(
                "Attached GPU ModelHost requires GPU-capable solver nodes.");
        }
        const auto path = resolve_attached_path(request);
        return {
            HostPlacement::attached,
            path,
            request.solver_nodes,
            true,
            true,
            true,
            path == DataPath::shared_memory
                ? "one ModelHost per solver node; SHM bulk path with UDS control"
                : path == DataPath::unix_socket
                      ? "one ModelHost per solver node; UDS data and control"
                      : "one ModelHost per solver node; explicit network data path",
        };
    }

    const auto path = resolve_central_path(request);
    return {
        HostPlacement::central,
        path,
        request.central_host_nodes,
        false,
        false,
        true,
        path == DataPath::ucx
            ? "central ModelHost nodes; UCX data path with TCP control"
            : "central ModelHost nodes; portable TCP data and control",
    };
}

const char* name(HostPlacement placement) noexcept {
    switch (placement) {
        case HostPlacement::automatic:
            return "auto";
        case HostPlacement::attached:
            return "attached";
        case HostPlacement::central:
            return "central";
    }
    return "unknown";
}

const char* name(InferenceDevice device) noexcept {
    return device == InferenceDevice::cpu ? "cpu" : "gpu";
}

const char* name(DataPath data_path) noexcept {
    switch (data_path) {
        case DataPath::shared_memory:
            return "shm";
        case DataPath::unix_socket:
            return "uds";
        case DataPath::ucx:
            return "ucx";
        case DataPath::tcp:
            return "tcp";
    }
    return "unknown";
}

}  // namespace foamnordic::native
