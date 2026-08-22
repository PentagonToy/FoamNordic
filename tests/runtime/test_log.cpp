#include <sstream>
#include <stdexcept>
#include <string>

#include "foamnordic/runtime/log.hpp"

int main() {
    std::ostringstream plain;
    foamnordic::native::log_to(
        plain, foamnordic::native::LogLevel::info, "Closure exchange ready.", false);
    if (plain.str() != "[FoamNord] Info: Closure exchange ready.\n") {
        throw std::runtime_error("Plain FoamNord log format is incorrect.");
    }

    std::ostringstream colored;
    foamnordic::native::log_to(
        colored, foamnordic::native::LogLevel::warning, "Timed handshake.", true);
    if (colored.str().find("\033[33m[FoamNord]\033[0m") == std::string::npos) {
        throw std::runtime_error("Colored FoamNord tag is incorrect.");
    }
}
