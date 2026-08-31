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
    [[nodiscard]] virtual bool supports(closure::ModelFormat format) const noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<closure::PackedModelKernel> load(
        const std::filesystem::path& payload,
        const closure::ModelArtifact& artifact) const = 0;
};

}  // namespace foamnordic::backend
