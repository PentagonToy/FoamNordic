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
#include <string>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/runner.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace foamnordic::inference {

struct WorkerOptions {
    bool shared_memory{true};
    bool ucx{false};
    std::string ucx_host;
    std::uint32_t connections{1};
    std::uint32_t model_threads{1};
    std::uint64_t maximum_payload{16ULL * 1024ULL * 1024ULL * 1024ULL};

    void validate() const;
};

class ModelWorker {
public:
    ModelWorker(
        fjord::FjordAddress address,
        ModelArtifact artifact,
        const CellEvaluationPolicy& bypass,
        ModelKernel& kernel,
        WorkerOptions options = {});

    ModelWorker(
        fjord::FjordAddress address,
        const std::filesystem::path& manifest_path,
        const CellEvaluationPolicy& bypass,
        WorkerOptions options = {});

    ModelWorker(const ModelWorker&) = delete;
    ModelWorker& operator=(const ModelWorker&) = delete;

    [[nodiscard]] fjord::FjordAddress address() const;
    void run();

private:
    fjord::FjordAddress requested_address_;
    fjord::FjordListener listener_;
    ModelArtifact artifact_;
    const CellEvaluationPolicy& bypass_;
    std::unique_ptr<ModelKernel> owned_kernel_;
    ModelKernel* kernel_;
    WorkerOptions options_;
};

}  // namespace foamnordic::inference
