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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
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
        std::span<const std::byte> payload,
        const inference::ModelArtifact& artifact,
        std::uint32_t threads) const = 0;
};

}  // namespace foamnordic::backend
