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

#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include <unistd.h>

#include "foamnordic/backend/inference/closure.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace {

void usage() {
    std::cerr
        << "Usage: foamnordic_closure_worker <address> <manifest> [OPTIONS]\n"
        << "  address   unix:///path/to/worker.sock or tcp://host:port\n"
        << "  manifest  FoamNordic native model manifest\n"
        << "  --connections N  Solver sessions accepted by this host\n"
        << "  --ready-file PATH Create PATH after the worker is ready\n"
        << "  --ucx-host HOST   Require UCX and advertise HOST to solvers\n"
        << "  --no-shm          Keep a Unix session on UDS\n";
}

std::uint32_t positive_count(std::string_view value) {
    std::uint32_t count = 0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), count);
    if (value.empty() || result.ec != std::errc{}
        || result.ptr != value.data() + value.size() || count == 0) {
        throw std::invalid_argument("--connections requires a positive integer.");
    }
    return count;
}

std::string expand_rank(std::string value) {
    constexpr std::string_view marker("{rank}");
    if (value.find(marker) == std::string::npos) {
        return value;
    }
    const char* rank = nullptr;
    for (const auto* variable : {
             "SLURM_PROCID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"}) {
        rank = std::getenv(variable);
        if (rank != nullptr && *rank != '\0') {
            break;
        }
    }
    if (rank == nullptr || *rank == '\0') {
        throw std::runtime_error(
            "A {rank} worker path requires an MPI or Slurm rank variable.");
    }
    std::size_t position = 0;
    while ((position = value.find(marker, position)) != std::string::npos) {
        value.replace(position, marker.size(), rank);
        position += std::char_traits<char>::length(rank);
    }
    return value;
}

class ReadyMarker {
public:
    explicit ReadyMarker(std::filesystem::path path) : path_(std::move(path)) {
        if (path_.empty()) {
            return;
        }
        const auto descriptor = ::open(
            path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0660);
        if (descriptor < 0) {
            throw std::system_error(
                errno,
                std::generic_category(),
                "Cannot create ClosureHost readiness marker");
        }
        const auto identity = std::to_string(static_cast<long long>(::getpid())) + "\n";
        const auto written = ::write(descriptor, identity.data(), identity.size());
        const auto saved_error = errno;
        ::close(descriptor);
        if (written != static_cast<ssize_t>(identity.size())) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
            throw std::system_error(
                saved_error == 0 ? EIO : saved_error,
                std::generic_category(),
                "Cannot write ClosureHost readiness marker");
        }
    }

    ~ReadyMarker() {
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

    ReadyMarker(const ReadyMarker&) = delete;
    ReadyMarker& operator=(const ReadyMarker&) = delete;

private:
    std::filesystem::path path_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    foamnordic::closure::WorkerOptions options;
    std::filesystem::path ready_file;
    try {
        for (int index = 3; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--no-shm") {
                options.shared_memory = false;
            } else if (argument == "--connections" && index + 1 < argc) {
                options.connections = positive_count(argv[++index]);
            } else if (argument == "--ready-file" && index + 1 < argc) {
                ready_file = expand_rank(argv[++index]);
            } else if (argument == "--ucx-host" && index + 1 < argc) {
                options.ucx = true;
                options.ucx_host = argv[++index];
            } else {
                throw std::invalid_argument(
                    "Unknown or incomplete closure worker option: "
                    + std::string(argument));
            }
        }
        foamnordic::closure::EvaluateEveryCell bypass;
        foamnordic::closure::NativeClosureWorker worker(
            foamnordic::fjord::FjordAddress::parse(argv[1]),
            std::filesystem::path(argv[2]),
            bypass,
            options);
        std::cout << "[FoamNordic] Closure worker ready: "
                  << worker.address().text() << std::endl;
        ReadyMarker ready(std::move(ready_file));
        worker.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FoamNordic] Closure worker failed: " << error.what()
                  << '\n';
        return 1;
    }
}
