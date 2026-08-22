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

#include <filesystem>
#include <memory>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/runner.hpp"

namespace foamnordic::closure {

struct LoadedModel {
    ModelArtifact artifact;
    std::unique_ptr<ModelKernel> kernel;
};

[[nodiscard]] LoadedModel load_model(
    const std::filesystem::path& manifest_path);

}  // namespace foamnordic::closure
