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

namespace foamnordic::inference {

struct ModelLoadOptions {
    std::uint32_t threads{1};
    std::uint32_t execution_contexts{1};
    std::uint64_t maximum_working_bytes{16ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t profile_every{0};

    void validate() const;
};

struct LoadedModel {
    ModelArtifact artifact;
    std::unique_ptr<ModelKernel> kernel;
};

[[nodiscard]] LoadedModel load_model(
    const std::filesystem::path& manifest_path,
    ModelLoadOptions options = {});

}  // namespace foamnordic::inference
