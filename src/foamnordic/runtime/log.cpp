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

#include "foamnordic/runtime/log.hpp"

#include <cstdio>
#include <iostream>
#include <mutex>
#include <string_view>

#include <unistd.h>

namespace foamnordic::native {
namespace {

std::mutex log_mutex;

std::string_view level_name(LogLevel level) {
    switch (level) {
        case LogLevel::debug:
            return "Debug";
        case LogLevel::info:
            return "Info";
        case LogLevel::warning:
            return "Warning";
        case LogLevel::error:
            return "Error";
    }
    return "Info";
}

}  // namespace

void log_to(std::ostream& stream, LogLevel level, std::string_view message, bool color) {
    std::scoped_lock lock(log_mutex);
    if (color) {
        stream << "\033[33m[FoamNord]\033[0m";
    } else {
        stream << "[FoamNord]";
    }
    stream << ' ' << level_name(level) << ": " << message << '\n';
}

void log(LogLevel level, std::string_view message) {
    log_to(std::clog, level, message, ::isatty(::fileno(stderr)) == 1);
}

}  // namespace foamnordic::native
