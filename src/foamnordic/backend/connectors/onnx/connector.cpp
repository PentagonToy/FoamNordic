#include "foamnordic/backend/connectors/onnx/connector.hpp"

#include <memory>

#include "foamnordic/backend/inference/onnx.hpp"

namespace foamnordic::backend {
namespace {

class OnnxConnector final : public ModelConnector {
public:
    std::string_view id() const noexcept override { return "onnx"; }

    bool supports(inference::ModelFormat format) const noexcept override {
        return format == inference::ModelFormat::onnx;
    }

    std::unique_ptr<inference::PackedModelKernel> load(
        const std::filesystem::path& payload,
        const inference::ModelArtifact&) const override {
        return std::make_unique<inference::OnnxPackedKernel>(payload);
    }
};

}  // namespace

std::unique_ptr<ModelConnector> make_onnx_connector() {
    return std::make_unique<OnnxConnector>();
}

}  // namespace foamnordic::backend
