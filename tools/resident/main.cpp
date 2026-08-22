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

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "foamnordic/backend/inference/closure.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace {

void usage() {
    std::cerr
        << "Usage: foamnordic_closure_worker <address> <manifest> [--no-shm]\n"
        << "  address   unix:///path/to/worker.sock or tcp://host:port\n"
        << "  manifest  FoamNordic native model manifest\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        usage();
        return 2;
    }
    foamnordic::closure::WorkerOptions options;
    if (argc == 4) {
        if (std::string(argv[3]) != "--no-shm") {
            usage();
            return 2;
        }
        options.shared_memory = false;
    }

    try {
        foamnordic::closure::EvaluateEveryCell bypass;
        foamnordic::closure::NativeClosureWorker worker(
            foamnordic::fjord::FjordAddress::parse(argv[1]),
            std::filesystem::path(argv[2]),
            bypass,
            options);
        std::cout << "[FoamNord] Closure worker ready: "
                  << worker.address().text() << std::endl;
        worker.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FoamNord] Closure worker failed: " << error.what()
                  << '\n';
        return 1;
    }
}
