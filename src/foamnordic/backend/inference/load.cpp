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

#include <memory>
#include <stdexcept>
#include <utility>

#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"

#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
#include "foamnordic/backend/inference/onnx.hpp"
#endif

namespace foamnordic::closure {
namespace {

class OwnedArtifactKernel final : public ModelKernel {
public:
    OwnedArtifactKernel(
        ModelArtifact artifact,
        std::unique_ptr<PackedModelKernel> packed)
        : packed_(std::move(packed)), kernel_(std::move(artifact), *packed_) {}

    TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time) override {
        return kernel_.evaluate(
            inputs, active_cells, exchange_index, physical_time);
    }

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

LoadedModel load_model(const std::filesystem::path& manifest_path) {
    auto artifact = read_manifest(manifest_path);
    std::unique_ptr<PackedModelKernel> packed;

    switch (artifact.format) {
        case ModelFormat::onnx:
#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
            packed = std::make_unique<OnnxPackedKernel>(
                resolve_artifact(manifest_path, artifact.artifact_path));
            break;
#else
            throw std::runtime_error(
                "This FoamNordic build does not include ONNX Runtime support.");
#endif
        case ModelFormat::equinox:
            throw std::runtime_error(
                "Native Equinox loading is not implemented yet.");
        case ModelFormat::joblib:
            throw std::runtime_error(
                "Native Joblib loading is not implemented yet.");
    }

    auto kernel =
        std::make_unique<OwnedArtifactKernel>(artifact, std::move(packed));
    return {std::move(artifact), std::move(kernel)};
}

}  // namespace foamnordic::closure
