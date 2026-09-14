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

#include "foamnordic/backend/inference/load.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <fstream>
#include "foamnordic/backend/inference/forest.hpp"
#include "bindnordic/table.hpp"

#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"
#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
#include "foamnordic/backend/inference/onnx.hpp"
#endif
#include "foamnordic/backend/connectors/registry.hpp"

namespace foamnordic::inference {
namespace {

class OwnedArtifactKernel final : public ModelKernel {
public:
    OwnedArtifactKernel(
        ModelArtifact artifact,
        std::unique_ptr<PackedModelKernel> packed,
        std::shared_ptr<const void> payload_owner,
        ExecutionResources resources)
        : payload_owner_(std::move(payload_owner)),
          packed_(std::move(packed)),
          kernel_(std::move(artifact), *packed_, resources) {}

    TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank) override {
        return kernel_.evaluate(
            inputs, active_cells, exchange_index, physical_time, rank);
    }

    [[nodiscard]] bool owns_backend_synchronization() const noexcept override {
        return kernel_.owns_backend_synchronization();
    }

    [[nodiscard]] ExecutionSnapshot execution_snapshot() const override {
        return kernel_.execution_snapshot();
    }

    [[nodiscard]] bool telemetry_enabled() const noexcept override {
        return kernel_.telemetry_enabled();
    }

    void observe_exchange(
        std::uint64_t rows,
        std::uint64_t active_cells,
        std::uint64_t input_payload_bytes,
        std::uint64_t output_payload_bytes) noexcept override {
        kernel_.observe_exchange(
            rows, active_cells, input_payload_bytes, output_payload_bytes);
    }

    void cancel_pending() noexcept override { kernel_.cancel_pending(); }

private:
    std::shared_ptr<const void> payload_owner_;
    std::unique_ptr<PackedModelKernel> packed_;
    ArtifactModelKernel kernel_;
};

std::filesystem::path resolve_artifact(
    const std::filesystem::path& manifest_path,
    const std::string& artifact_path) {
    std::filesystem::path resolved(artifact_path);
    if (resolved.is_relative()) {
        resolved = manifest_path.parent_path() / resolved;
    }
    return std::filesystem::weakly_canonical(resolved);
}

}  // namespace

void ModelLoadOptions::validate() const {
    if (threads == 0
        || threads > static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Model thread count must be positive.");
    }
    ExecutionResources{
        execution_contexts,
        maximum_working_bytes,
        profile_every,
        capacity,
        capacity_units,
    }.validate();
}

LoadedModel load_model(
    const std::filesystem::path& manifest_path,
    ModelLoadOptions options) {
    options.validate();
    auto artifact = read_manifest(manifest_path);
    std::unique_ptr<PackedModelKernel> packed;
    std::shared_ptr<const void> payload_owner;
    if (artifact.format == ModelFormat::table) {
        if (!is_bundle(manifest_path)) {
            throw std::invalid_argument("Table models require a self-contained FNOM bundle.");
        }
        std::uint64_t inputs = 0;
        std::uint64_t outputs = 0;
        for (const auto& field : artifact.contract.inputs) inputs += field.components;
        for (const auto& field : artifact.contract.outputs) outputs += field.components;
        auto mapped = map_bundle_payload(manifest_path);
        packed = std::make_unique<bindnordic::TableKernel>(
            mapped->bytes(), inputs, outputs,
            artifact.contract.inputs.front().element);
        payload_owner = std::move(mapped);
    } else if (artifact.runtime == "forest") {
        std::span<const std::byte> data;
        std::shared_ptr<std::vector<std::byte>> buffered;
        if (is_bundle(manifest_path)) {
            auto mapped = map_bundle_payload(manifest_path);
            data = mapped->bytes();
            payload_owner = std::move(mapped);
        }
        else {
            std::ifstream stream(resolve_artifact(manifest_path, artifact.artifact_path), std::ios::binary | std::ios::ate);
            if (!stream || stream.tellg() < 0) throw std::runtime_error("Cannot read forest payload.");
            buffered = std::make_shared<std::vector<std::byte>>(
                static_cast<std::size_t>(stream.tellg()));
            stream.seekg(0);
            if (!stream.read(reinterpret_cast<char*>(buffered->data()), buffered->size()))
                throw std::runtime_error("Truncated forest payload.");
            data = *buffered;
            payload_owner = std::move(buffered);
        }
        packed = std::make_unique<ForestKernel>(data, options.threads);
    } else
#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
    if (artifact.format == ModelFormat::onnx && is_bundle(manifest_path)) {
        auto mapped = map_bundle_payload(manifest_path);
        packed = std::make_unique<OnnxPackedKernel>(
            mapped->bytes(),
            OnnxOptions{static_cast<std::int32_t>(options.threads), 1});
        payload_owner = std::move(mapped);
    } else
#endif
    {
        std::vector<std::byte> data;
        if (is_bundle(manifest_path)) {
            auto mapped = map_bundle_payload(manifest_path);
            data.assign(mapped->bytes().begin(), mapped->bytes().end());
        } else {
            std::ifstream stream(
                resolve_artifact(manifest_path, artifact.artifact_path),
                std::ios::binary | std::ios::ate);
            if (!stream || stream.tellg() < 0) {
                throw std::runtime_error("Cannot read model connector payload.");
            }
            data.resize(static_cast<std::size_t>(stream.tellg()));
            stream.seekg(0);
            if (!stream.read(reinterpret_cast<char*>(data.data()), data.size())) {
                throw std::runtime_error("Truncated model connector payload.");
            }
        }
        packed = foamnordic::backend::connect_model(data, artifact, options.threads);
    }

    auto kernel = std::make_unique<OwnedArtifactKernel>(
        artifact,
        std::move(packed),
        std::move(payload_owner),
        ExecutionResources{
            options.execution_contexts,
            options.maximum_working_bytes,
            options.profile_every,
            std::move(options.capacity),
            options.capacity_units,
        });
    return {std::move(artifact), std::move(kernel)};
}

}  // namespace foamnordic::inference
