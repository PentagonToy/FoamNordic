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
        ExecutionResources resources)
        : packed_(std::move(packed)),
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

    void cancel_pending() noexcept override { kernel_.cancel_pending(); }

private:
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
    ExecutionResources{execution_contexts, maximum_working_bytes}.validate();
}

LoadedModel load_model(
    const std::filesystem::path& manifest_path,
    ModelLoadOptions options) {
    options.validate();
    auto artifact = read_manifest(manifest_path);
    std::unique_ptr<PackedModelKernel> packed;
    if (artifact.format == ModelFormat::table) {
        if (!is_bundle(manifest_path)) {
            throw std::invalid_argument("Table models require a self-contained FNOM bundle.");
        }
        std::uint64_t inputs = 0;
        std::uint64_t outputs = 0;
        for (const auto& field : artifact.contract.inputs) inputs += field.components;
        for (const auto& field : artifact.contract.outputs) outputs += field.components;
        packed = std::make_unique<bindnordic::TableKernel>(
            read_bundle_payload(manifest_path), inputs, outputs,
            artifact.contract.inputs.front().element);
    } else if (artifact.runtime == "forest") {
        std::vector<std::byte> data;
        if (is_bundle(manifest_path)) data = read_bundle_payload(manifest_path);
        else {
            std::ifstream stream(resolve_artifact(manifest_path, artifact.artifact_path), std::ios::binary | std::ios::ate);
            if (!stream || stream.tellg() < 0) throw std::runtime_error("Cannot read forest payload.");
            data.resize(static_cast<std::size_t>(stream.tellg()));
            stream.seekg(0);
            if (!stream.read(reinterpret_cast<char*>(data.data()), data.size()))
                throw std::runtime_error("Truncated forest payload.");
        }
        packed = std::make_unique<ForestKernel>(data, options.threads);
    } else
#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
    if (artifact.format == ModelFormat::onnx && is_bundle(manifest_path)) {
        packed = std::make_unique<OnnxPackedKernel>(
            read_bundle_payload(manifest_path),
            OnnxOptions{static_cast<std::int32_t>(options.threads), 1});
    } else
#endif
    {
        packed = foamnordic::backend::connect_model(
            resolve_artifact(manifest_path, artifact.artifact_path), artifact);
    }

    auto kernel = std::make_unique<OwnedArtifactKernel>(
        artifact,
        std::move(packed),
        ExecutionResources{
            options.execution_contexts,
            options.maximum_working_bytes,
            options.profile_every,
        });
    return {std::move(artifact), std::move(kernel)};
}

}  // namespace foamnordic::inference
