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
#include <string>
#include <vector>

#include "foamnordic/backend/connectors/connector.hpp"

namespace foamnordic::backend {

void register_connector(std::unique_ptr<ModelConnector> connector);

[[nodiscard]] std::unique_ptr<inference::PackedModelKernel> connect_model(
    const std::filesystem::path& payload,
    const inference::ModelArtifact& artifact);

[[nodiscard]] std::vector<std::string> available_connectors();

}  // namespace foamnordic::backend
