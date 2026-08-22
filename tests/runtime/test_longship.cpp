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

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "foamnordic/runtime/longship.hpp"
#include "foamnordic/runtime/longship_supervisor.hpp"

#include <unistd.h>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_single_node_longship() {
    foamnordic::native::LongshipRequest request;
    request.name = "lidDrivenCavity";
    request.solver_tasks = 16;
    request.host_cpus_per_node = 4;
    const auto plan = foamnordic::native::plan_longship(request);

    require(plan.allocation_nodes == 1, "Single-node allocation is invalid.");
    require(plan.solver_tasks_per_node == 16, "Solver layout is invalid.");
    require(plan.host_tasks == 1, "ClosureHost was not attached.");
    require(plan.allocation_cpus_per_node == 20, "Host CPUs were not reserved.");
    require(plan.host_starts_first && plan.fail_together, "Lifecycle is not coupled.");
    require(
        plan.placement.data_path
            == foamnordic::native::DataPath::shared_memory,
        "Attached Longship did not select SHM.");
}

void test_multi_node_longship() {
    foamnordic::native::LongshipRequest request;
    request.solver_nodes = 4;
    request.solver_tasks = 64;
    request.solver_cpus_per_task = 2;
    request.host_cpus_per_node = 3;
    const auto plan = foamnordic::native::plan_longship(request);

    require(plan.host_tasks == 4, "Longship requires one host per node.");
    require(plan.solver_tasks_per_node == 16, "Ranks were distributed incorrectly.");
    require(plan.allocation_cpus_per_node == 35, "Per-node CPUs are incorrect.");
}

void test_uneven_solver_layout_is_rejected() {
    foamnordic::native::LongshipRequest request;
    request.solver_nodes = 3;
    request.solver_tasks = 16;
    bool rejected = false;
    try {
        static_cast<void>(foamnordic::native::plan_longship(request));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Longship accepted an uneven rank layout.");
}

void test_central_host_is_not_silently_attached() {
    foamnordic::native::LongshipRequest request;
    request.placement.placement = foamnordic::native::HostPlacement::central;
    bool rejected = false;
    try {
        static_cast<void>(foamnordic::native::plan_longship(request));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Longship silently accepted a central host.");
}

std::filesystem::path readiness_path(const std::string& suffix) {
    return std::filesystem::temp_directory_path()
           / ("foamnordic-longship-"
              + std::to_string(static_cast<long long>(::getpid())) + '-'
              + suffix + ".ready");
}

foamnordic::native::LongshipCommand shell_command(std::string command) {
    return {{"/bin/sh", "-c", std::move(command)}, {}};
}

void test_supervisor_completes_with_solver() {
    const auto ready = readiness_path("complete");
    const auto launch = foamnordic::native::LongshipLaunch{
        shell_command(
            "test ! -e " + ready.string() + " && touch " + ready.string()
            + "; while :; do sleep 1; done"),
        shell_command("exit 0"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(50),
    };
    const auto result = foamnordic::native::sail_longship(launch);
    require(result.success(), "Longship rejected a successful solver.");
    std::filesystem::remove(ready);
}

void test_supervisor_propagates_solver_failure() {
    const auto ready = readiness_path("solver-failure");
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; while :; do sleep 1; done"),
        shell_command("exit 7"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(50),
    });
    require(result.solver_status == 7, "Solver failure status was lost.");
    require(!result.success(), "Failed solver produced a successful Longship.");
    std::filesystem::remove(ready);
}

void test_supervisor_propagates_host_failure() {
    const auto ready = readiness_path("host-failure");
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; sleep 0.05; exit 9"),
        shell_command("sleep 5"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(50),
    });
    require(result.host_failed_first, "Early ClosureHost exit was not detected.");
    require(result.host_status == 9, "ClosureHost failure status was lost.");
    require(!result.success(), "Failed ClosureHost produced a successful Longship.");
    std::filesystem::remove(ready);
}

void test_supervisor_times_out_before_solver_start() {
    const auto ready = readiness_path("timeout");
    bool timed_out = false;
    try {
        static_cast<void>(foamnordic::native::sail_longship({
            shell_command("sleep 5"),
            shell_command("exit 0"),
            {ready},
            std::chrono::milliseconds(30),
            std::chrono::milliseconds(50),
        }));
    } catch (const std::runtime_error&) {
        timed_out = true;
    }
    require(timed_out, "Longship accepted an unready ClosureHost.");
}

}  // namespace

int main() {
    test_single_node_longship();
    test_multi_node_longship();
    test_uneven_solver_layout_is_rejected();
    test_central_host_is_not_silently_attached();
    test_supervisor_completes_with_solver();
    test_supervisor_propagates_solver_failure();
    test_supervisor_propagates_host_failure();
    test_supervisor_times_out_before_solver_start();
}
