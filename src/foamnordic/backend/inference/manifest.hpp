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
#include <span>
#include <vector>

#include "foamnordic/backend/inference/artifact.hpp"

namespace foamnordic::inference {

struct BundlePayloadRegion {
    std::uint64_t offset;
    std::uint64_t size;
};

[[nodiscard]] std::vector<std::byte> encode_manifest(
    const ModelArtifact& artifact);

[[nodiscard]] ModelArtifact decode_manifest(
    std::span<const std::byte> bytes);

void write_manifest(
    const std::filesystem::path& path,
    const ModelArtifact& artifact);

void write_bundle(
    const std::filesystem::path& path,
    const ModelArtifact& artifact,
    const std::filesystem::path& payload_path);

void extract_bundle_payload(
    const std::filesystem::path& path,
    const std::filesystem::path& destination);

[[nodiscard]] BundlePayloadRegion bundle_payload_region(
    const std::filesystem::path& path);

[[nodiscard]] ModelArtifact read_manifest(
    const std::filesystem::path& path);

[[nodiscard]] bool is_bundle(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::byte> read_bundle_payload(
    const std::filesystem::path& path);

}  // namespace foamnordic::inference
