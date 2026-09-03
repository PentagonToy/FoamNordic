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
#include <string_view>
#include <thread>
#include <vector>

#include "foamnordic/runtime/longship.hpp"
#include "foamnordic/runtime/longship_cli.hpp"
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
    require(plan.host_tasks == 1, "ModelHost was not attached.");
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

void test_solver_only_longship() {
    foamnordic::native::LongshipRequest request;
    request.solver_tasks = 8;
    request.solver_cpus_per_task = 2;
    request.use_model_host = false;
    request.host_cpus_per_node = 0;
    const auto plan = foamnordic::native::plan_longship(request);

    require(plan.host_tasks == 0, "Solver-only plan allocated a ModelHost.");
    require(plan.host_cpus_per_task == 0, "Solver-only plan reserved host CPUs.");
    require(plan.allocation_cpus_per_node == 16, "Solver-only CPUs are incorrect.");
    require(!plan.host_starts_first, "Solver-only plan has a host startup phase.");
    require(!plan.fail_together, "Solver-only plan claims coupled components.");
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

void test_longship_cli_preserves_component_arguments() {
    const std::vector<std::string_view> arguments{
        "--ready",
        "/tmp/host-0.ready",
        "--ready",
        "/tmp/host-1.ready",
        "--readiness-timeout-ms",
        "45000",
        "--termination-grace-ms",
        "3000",
        "--host-output",
        "host.log",
        "--solver-output",
        "solver.log",
        "--host",
        "srun",
        "--nodes=2",
        "foamnordic_model_worker",
        "--solver",
        "srun",
        "--ntasks=32",
        "pimpleFoam",
        "-parallel",
    };
    const auto request = foamnordic::native::parse_longship_arguments(arguments);
    require(
        request.launch.host_ready_files.size() == 2
            && request.launch.readiness_timeout == std::chrono::seconds(45)
            && request.launch.termination_grace == std::chrono::seconds(3),
        "Longship CLI changed lifecycle options.");
    require(
        request.launch.host.arguments
                == std::vector<std::string>{
                    "srun", "--nodes=2", "foamnordic_model_worker"}
            && request.launch.solver.arguments
                   == std::vector<std::string>{
                       "srun", "--ntasks=32", "pimpleFoam", "-parallel"},
        "Longship CLI changed component arguments.");
    require(
        request.launch.host.output == "host.log"
            && request.launch.solver.output == "solver.log",
        "Longship CLI changed output paths.");
}

void test_longship_cli_rejects_incomplete_launch() {
    const std::vector<std::string_view> arguments{
        "--ready", "/tmp/host.ready", "--host", "host-only"};
    bool rejected = false;
    try {
        static_cast<void>(foamnordic::native::parse_longship_arguments(arguments));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Longship CLI accepted a launch without a solver.");
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

void test_supervisor_removes_readiness_after_forced_host_stop() {
    const auto ready = readiness_path("cleanup");
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; while :; do sleep 1; done"),
        shell_command("exit 0"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(20),
    });
    require(result.success(), "Forced host cleanup changed solver success.");
    require(
        !std::filesystem::exists(ready),
        "Longship left its readiness marker after host termination.");
}

void test_supervisor_signals_resident_host_before_grace_expires() {
    const auto ready = readiness_path("prompt-host-stop");
    const auto started = std::chrono::steady_clock::now();
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; while :; do sleep 1; done"),
        shell_command("exit 0"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::seconds(1),
    });
    const auto elapsed = std::chrono::steady_clock::now() - started;

    require(result.success(), "Prompt host shutdown changed solver success.");
    require(
        elapsed < std::chrono::milliseconds(500),
        "Longship delayed SIGTERM until the host grace period expired.");
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
    require(result.host_failed_first, "Early ModelHost exit was not detected.");
    require(result.host_status == 9, "ModelHost failure status was lost.");
    require(!result.success(), "Failed ModelHost produced a successful Longship.");
    std::filesystem::remove(ready);
}

void test_supervisor_accepts_protocol_host_shutdown_before_solver_exit() {
    const auto ready = readiness_path("protocol-shutdown");
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; sleep 0.02; exit 0"),
        shell_command("sleep 0.04; exit 0"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(100),
    });
    require(
        result.success(),
        "Longship rejected a host that shut down during solver finalization.");
}

void test_supervisor_rejects_early_clean_host_exit() {
    const auto ready = readiness_path("early-clean-host");
    const auto result = foamnordic::native::sail_longship({
        shell_command("touch " + ready.string() + "; exit 0"),
        shell_command("sleep 5"),
        {ready},
        std::chrono::seconds(1),
        std::chrono::milliseconds(20),
    });
    require(
        result.host_failed_first,
        "Longship accepted a clean host exit while the solver remained active.");
    require(!result.success(), "Early host exit produced a successful Longship.");
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
    require(timed_out, "Longship accepted an unready ModelHost.");
}

void test_supervisor_cancels_components_together() {
    const auto ready = readiness_path("cancel");
    foamnordic::native::LongshipStop stop;
    std::thread cancellation([&stop] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        stop.request_stop();
    });
    const auto result = foamnordic::native::sail_longship(
        {
            shell_command("touch " + ready.string() + "; exec sleep 30"),
            shell_command("exec sleep 30"),
            {ready},
            std::chrono::seconds(2),
            std::chrono::milliseconds(100),
        },
        &stop);
    cancellation.join();
    require(result.cancelled, "Longship did not report external cancellation.");
    require(!result.success(), "Cancelled Longship reported success.");
    require(
        !std::filesystem::exists(ready),
        "Cancelled Longship left its readiness marker.");
}

}  // namespace

int main() {
    test_single_node_longship();
    test_multi_node_longship();
    test_solver_only_longship();
    test_uneven_solver_layout_is_rejected();
    test_central_host_is_not_silently_attached();
    test_longship_cli_preserves_component_arguments();
    test_longship_cli_rejects_incomplete_launch();
    test_supervisor_completes_with_solver();
    test_supervisor_removes_readiness_after_forced_host_stop();
    test_supervisor_signals_resident_host_before_grace_expires();
    test_supervisor_propagates_solver_failure();
    test_supervisor_propagates_host_failure();
    test_supervisor_accepts_protocol_host_shutdown_before_solver_exit();
    test_supervisor_rejects_early_clean_host_exit();
    test_supervisor_times_out_before_solver_start();
    test_supervisor_cancels_components_together();
}
