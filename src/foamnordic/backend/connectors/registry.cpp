#include "foamnordic/backend/connectors/registry.hpp"

#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
#include "foamnordic/backend/connectors/onnx/connector.hpp"
#endif

namespace foamnordic::backend {
namespace {

struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<ModelConnector>> connectors;
};

Registry& registry() {
    static Registry value;
    static const bool initialized = [] {
#ifdef FOAMNORDIC_HAS_ONNXRUNTIME
        value.connectors.push_back(make_onnx_connector());
#endif
        return true;
    }();
    static_cast<void>(initialized);
    return value;
}

}  // namespace

void register_connector(std::unique_ptr<ModelConnector> connector) {
    if (!connector || connector->id().empty()) {
        throw std::invalid_argument(
            "FoamNordic backend connector requires a stable non-empty id.");
    }
    auto& value = registry();
    std::scoped_lock lock(value.mutex);
    for (const auto& existing : value.connectors) {
        if (existing->id() == connector->id()) {
            throw std::invalid_argument(
                "FoamNordic backend connector id is already registered: "
                + std::string(connector->id()));
        }
    }
    value.connectors.push_back(std::move(connector));
}

std::unique_ptr<inference::PackedModelKernel> connect_model(
    const std::filesystem::path& payload,
    const inference::ModelArtifact& artifact) {
    auto& value = registry();
    std::scoped_lock lock(value.mutex);
    for (const auto& connector : value.connectors) {
        if (connector->supports(artifact.format)) {
            return connector->load(payload, artifact);
        }
    }
    throw std::runtime_error(
        "No registered FoamNordic backend connector supports model format: "
        + std::string(inference::name(artifact.format)));
}

std::vector<std::string> available_connectors() {
    auto& value = registry();
    std::scoped_lock lock(value.mutex);
    std::vector<std::string> result;
    result.reserve(value.connectors.size());
    for (const auto& connector : value.connectors) {
        result.emplace_back(connector->id());
    }
    return result;
}

}  // namespace foamnordic::backend
