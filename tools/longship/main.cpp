#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "foamnordic/runtime/log.hpp"
#include "foamnordic/runtime/longship_cli.hpp"

int main(int argc, char** argv) {
    try {
        std::vector<std::string_view> arguments;
        arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
        for (int index = 1; index < argc; ++index) {
            arguments.emplace_back(argv[index]);
        }
        const auto request = foamnordic::native::parse_longship_arguments(arguments);
        if (request.show_help) {
            std::cout << foamnordic::native::longship_usage();
        }
        return foamnordic::native::run_longship(request);
    } catch (const std::exception& error) {
        foamnordic::native::log(
            foamnordic::native::LogLevel::error, error.what());
        std::cerr << foamnordic::native::longship_usage();
        return 2;
    }
}
