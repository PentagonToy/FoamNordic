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
