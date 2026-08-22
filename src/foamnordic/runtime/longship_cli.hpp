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

#include <span>
#include <string>
#include <string_view>

#include "foamnordic/runtime/longship_supervisor.hpp"

namespace foamnordic::native {

struct LongshipCliRequest {
    LongshipLaunch launch;
    bool show_help{false};
};

[[nodiscard]] LongshipCliRequest parse_longship_arguments(
    std::span<const std::string_view> arguments);
[[nodiscard]] std::string longship_usage();
[[nodiscard]] int run_longship(
    const LongshipCliRequest& request,
    const LongshipStop* stop = nullptr);

}  // namespace foamnordic::native
