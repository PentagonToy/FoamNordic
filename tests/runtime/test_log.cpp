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
#include <sstream>
#include <stdexcept>
#include <string>

#include "foamnordic/runtime/log.hpp"

int main() {
    std::ostringstream plain;
    foamnordic::native::log_to(
        plain, foamnordic::native::LogLevel::info, "Closure exchange ready.", false);
    if (plain.str() != "[FoamNordic] Info: Closure exchange ready.\n") {
        throw std::runtime_error("Plain FoamNordic log format is incorrect.");
    }

    std::ostringstream colored;
    foamnordic::native::log_to(
        colored, foamnordic::native::LogLevel::warning, "Timed handshake.", true);
    if (colored.str().find("\033[33m[FoamNordic]\033[0m") == std::string::npos) {
        throw std::runtime_error("Colored FoamNordic tag is incorrect.");
    }
}
