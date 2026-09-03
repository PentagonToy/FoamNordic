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

#include "foamnordic/runtime/longship_cli.hpp"

#include <charconv>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "foamnordic/runtime/log.hpp"

namespace foamnordic::native {
namespace {

[[nodiscard]] std::chrono::milliseconds parse_milliseconds(
    std::string_view value,
    std::string_view option) {
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{}
        || result.ptr != value.data() + value.size() || parsed == 0
        || parsed > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            std::string(option) + " requires positive milliseconds.");
    }
    return std::chrono::milliseconds(parsed);
}

[[nodiscard]] std::string_view require_value(
    std::span<const std::string_view> arguments,
    std::size_t& index,
    std::string_view option) {
    ++index;
    if (index >= arguments.size() || arguments[index].empty()) {
        throw std::invalid_argument(
            std::string(option) + " requires a value.");
    }
    return arguments[index];
}

[[nodiscard]] std::vector<std::string> copy_command(
    std::span<const std::string_view> arguments) {
    std::vector<std::string> command;
    command.reserve(arguments.size());
    for (const auto argument : arguments) {
        command.emplace_back(argument);
    }
    return command;
}

[[nodiscard]] int exit_status(const LongshipResult& result) noexcept {
    if (result.cancelled) {
        return 130;
    }
    const auto status = result.host_failed_first
                            ? result.host_status
                            : result.solver_status;
    return status > 0 && status <= 255 ? status : 1;
}

}  // namespace

LongshipCliRequest parse_longship_arguments(
    std::span<const std::string_view> arguments) {
    LongshipCliRequest request;
    std::size_t host_marker = arguments.size();
    std::size_t solver_marker = arguments.size();

    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const auto argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            if (arguments.size() != 1) {
                throw std::invalid_argument(
                    "Longship help cannot be combined with launch arguments.");
            }
            request.show_help = true;
            return request;
        }
        if (argument == "--host") {
            host_marker = index;
            break;
        }
        if (argument == "--ready") {
            request.launch.host_ready_files.emplace_back(
                require_value(arguments, index, argument));
        } else if (argument == "--host-output") {
            request.launch.host.output = require_value(arguments, index, argument);
        } else if (argument == "--solver-output") {
            request.launch.solver.output = require_value(arguments, index, argument);
        } else if (argument == "--readiness-timeout-ms") {
            request.launch.readiness_timeout = parse_milliseconds(
                require_value(arguments, index, argument), argument);
        } else if (argument == "--termination-grace-ms") {
            request.launch.termination_grace = parse_milliseconds(
                require_value(arguments, index, argument), argument);
        } else {
            throw std::invalid_argument(
                "Unknown Longship option before --host: "
                + std::string(argument));
        }
    }

    if (host_marker == arguments.size()) {
        throw std::invalid_argument("Longship launch requires --host.");
    }
    for (std::size_t index = host_marker + 1; index < arguments.size(); ++index) {
        if (arguments[index] == "--solver") {
            solver_marker = index;
            break;
        }
    }
    if (solver_marker == arguments.size()) {
        throw std::invalid_argument("Longship launch requires --solver.");
    }

    request.launch.host.arguments = copy_command(
        arguments.subspan(host_marker + 1, solver_marker - host_marker - 1));
    request.launch.solver.arguments = copy_command(
        arguments.subspan(solver_marker + 1));
    request.launch.validate();
    return request;
}

std::string longship_usage() {
    return
        "Usage:\n"
        "  foamnordic-longship --ready PATH [--ready PATH ...] [OPTIONS]\n"
        "      --host COMMAND [ARG ...] --solver COMMAND [ARG ...]\n\n"
        "Options:\n"
        "  --host-output PATH          Redirect ModelHost stdout and stderr.\n"
        "  --solver-output PATH        Redirect solver stdout and stderr.\n"
        "  --readiness-timeout-ms N    Wait for all readiness files (default 30000).\n"
        "  --termination-grace-ms N    Grace before SIGKILL (default 2000).\n"
        "  -h, --help                  Show this help.\n\n"
        "Readiness paths may be regular marker files or Unix sockets.\n"
        "Command arguments are passed directly without an intermediate shell.\n";
}

int run_longship(
    const LongshipCliRequest& request,
    const LongshipStop* stop) {
    if (request.show_help) {
        return 0;
    }
    log(LogLevel::info, "Starting ModelHost component.");
    const auto result = sail_longship(request.launch, stop);
    if (result.success()) {
        log(LogLevel::info, "Longship completed successfully.");
        return 0;
    }
    if (result.cancelled) {
        log(LogLevel::warning, "Longship cancelled; components terminated together.");
    } else if (result.host_failed_first) {
        log(LogLevel::error, "ModelHost exited before the solver completed.");
    } else {
        log(LogLevel::error, "Solver exited with a failure status.");
    }
    return exit_status(result);
}

}  // namespace foamnordic::native
