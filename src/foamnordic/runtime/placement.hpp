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

namespace foamnordic::native {

enum class HostPlacement {
    automatic,
    attached,
    central,
};

enum class InferenceDevice {
    cpu,
    gpu,
};

enum class DataPathPreference {
    automatic,
    shared_memory,
    unix_socket,
    ucx,
    tcp,
};

enum class DataPath {
    shared_memory,
    unix_socket,
    ucx,
    tcp,
};

struct PlacementRequest {
    HostPlacement placement{HostPlacement::automatic};
    InferenceDevice device{InferenceDevice::cpu};
    DataPathPreference data_path{DataPathPreference::automatic};
    std::uint32_t solver_nodes{1};
    std::uint32_t central_host_nodes{1};
    bool solver_nodes_have_device{true};
    bool shared_memory_available{true};
    bool unix_socket_available{true};
    bool ucx_available{false};

    void validate() const;
};

struct PlacementPlan {
    HostPlacement placement{HostPlacement::attached};
    DataPath data_path{DataPath::unix_socket};
    std::uint32_t host_instances{1};
    bool same_allocation{true};
    bool same_node{true};
    bool coupled_lifetime{true};
    std::string reason;
};

[[nodiscard]] PlacementPlan resolve_placement(const PlacementRequest& request);
[[nodiscard]] const char* name(HostPlacement placement) noexcept;
[[nodiscard]] const char* name(InferenceDevice device) noexcept;
[[nodiscard]] const char* name(DataPath data_path) noexcept;

}  // namespace foamnordic::native
