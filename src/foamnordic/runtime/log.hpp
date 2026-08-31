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

#include <ostream>
#include <string_view>

namespace foamnordic::native {

enum class LogLevel {
    debug,
    info,
    warning,
    error,
};

void log(LogLevel level, std::string_view message);
void log_to(std::ostream& stream, LogLevel level, std::string_view message, bool color);

}  // namespace foamnordic::native
