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

#include <cstdint>
#include <filesystem>
#include <memory>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/runner.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace foamnordic::closure {

struct WorkerOptions {
    bool shared_memory{true};
    std::uint64_t maximum_payload{16ULL * 1024ULL * 1024ULL * 1024ULL};

    void validate() const;
};

class NativeClosureWorker {
public:
    NativeClosureWorker(
        fjord::FjordAddress address,
        ModelArtifact artifact,
        const BypassPolicy& bypass,
        ModelKernel& kernel,
        WorkerOptions options = {});

    NativeClosureWorker(
        fjord::FjordAddress address,
        const std::filesystem::path& manifest_path,
        const BypassPolicy& bypass,
        WorkerOptions options = {});

    NativeClosureWorker(const NativeClosureWorker&) = delete;
    NativeClosureWorker& operator=(const NativeClosureWorker&) = delete;

    [[nodiscard]] fjord::FjordAddress address() const;
    void run();

private:
    fjord::FjordAddress requested_address_;
    fjord::FjordListener listener_;
    ModelArtifact artifact_;
    const BypassPolicy& bypass_;
    std::unique_ptr<ModelKernel> owned_kernel_;
    ModelKernel* kernel_;
    WorkerOptions options_;
};

}  // namespace foamnordic::closure
