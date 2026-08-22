#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "foamnordic/runtime/log.hpp"
#include "foamnordic/runtime/longship_cli.hpp"

namespace {

volatile std::sig_atomic_t stop_signal = 0;

extern "C" void request_stop(int) { stop_signal = 1; }

}  // namespace

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
        foamnordic::native::LongshipStop stop;
        std::atomic<bool> finished{false};
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        std::thread signal_watcher([&] {
            while (!finished.load(std::memory_order_acquire)) {
                if (stop_signal != 0) {
                    stop.request_stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        try {
            const auto status = foamnordic::native::run_longship(request, &stop);
            finished.store(true, std::memory_order_release);
            signal_watcher.join();
            return status;
        } catch (...) {
            finished.store(true, std::memory_order_release);
            signal_watcher.join();
            throw;
        }
    } catch (const std::exception& error) {
        foamnordic::native::log(
            foamnordic::native::LogLevel::error, error.what());
        std::cerr << foamnordic::native::longship_usage();
        return 2;
    }
}
