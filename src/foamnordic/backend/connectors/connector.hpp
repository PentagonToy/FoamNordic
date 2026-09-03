#pragma once

#include <filesystem>
#include <memory>
#include <string_view>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/model.hpp"

namespace foamnordic::backend {

class ModelConnector {
public:
    virtual ~ModelConnector() = default;

    [[nodiscard]] virtual std::string_view id() const noexcept = 0;
    [[nodiscard]] virtual bool supports(inference::ModelFormat format) const noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<inference::PackedModelKernel> load(
        const std::filesystem::path& payload,
        const inference::ModelArtifact& artifact) const = 0;
};

}  // namespace foamnordic::backend
