#include "foamnordic/backend/connectors/plugin.hpp"

#include <array>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "foamnordic/backend/connectors/connector.hpp"
#include "foamnordic/backend/connectors/plugin.h"
#include "foamnordic/backend/connectors/registry.hpp"

namespace foamnordic::backend {
namespace {

class PluginLibrary {
public:
    explicit PluginLibrary(const std::filesystem::path& path) {
        handle_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            throw std::runtime_error(
                "Cannot load FoamNordic connector plugin: "
                + std::string(::dlerror()));
        }
        ::dlerror();
        const auto entry = reinterpret_cast<foamnordic_connector_entry_v1>(
            ::dlsym(handle_, "foamnordic_connector_v1"));
        if (const auto* error = ::dlerror(); error != nullptr || entry == nullptr) {
            ::dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error(
                "FoamNordic connector plugin has no v1 entry point: "
                + std::string(error == nullptr ? "missing symbol" : error));
        }
        api_ = entry();
        if (api_ == nullptr || api_->abi_version != FOAMNORDIC_CONNECTOR_ABI_V1
            || api_->id == nullptr || api_->id[0] == '\0' || api_->create == nullptr
            || api_->destroy == nullptr || api_->evaluate == nullptr
            || api_->release == nullptr) {
            ::dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error("FoamNordic connector plugin has an invalid v1 ABI.");
        }
    }

    ~PluginLibrary() {
        if (handle_ != nullptr) {
            ::dlclose(handle_);
        }
    }

    PluginLibrary(const PluginLibrary&) = delete;
    PluginLibrary& operator=(const PluginLibrary&) = delete;

    [[nodiscard]] const foamnordic_connector_api_v1& api() const noexcept {
        return *api_;
    }

private:
    void* handle_{nullptr};
    const foamnordic_connector_api_v1* api_{nullptr};
};

std::string plugin_error(const std::array<char, 1024>& error) {
    return error[0] == '\0' ? "connector operation failed" : std::string(error.data());
}

class PluginKernel final : public inference::PackedModelKernel {
public:
    PluginKernel(
        std::shared_ptr<PluginLibrary> library,
        std::span<const std::byte> payload,
        std::uint32_t threads)
        : library_(std::move(library)) {
        std::array<char, 1024> error{};
        model_ = library_->api().create(
            payload.data(), payload.size(), threads, error.data(), error.size());
        if (model_ == nullptr) {
            throw std::runtime_error(plugin_error(error));
        }
    }

    ~PluginKernel() override {
        if (model_ != nullptr) {
            library_->api().destroy(model_);
        }
    }

    [[nodiscard]] bindnordic::BackendCapabilities capabilities()
        const noexcept override {
        const auto& api = library_->api();
        return {
            api.concurrent_evaluation != 0,
            api.row_independent != 0,
            api.owns_compute_threads != 0,
        };
    }

    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank) override {
        features.validate();
        if (features.shape.size() != 2) {
            throw std::invalid_argument("Connector input tensor must be rank two.");
        }
        std::array<char, 1024> error{};
        foamnordic_connector_result_v1 result{};
        const auto status = library_->api().evaluate(
            model_, features.bytes.data(), features.bytes.size(),
            features.shape[0], features.shape[1],
            static_cast<std::uint8_t>(features.element), exchange_index,
            physical_time, rank, &result, error.data(), error.size());
        if (status != 0) {
            throw std::runtime_error(plugin_error(error));
        }
        struct Release {
            const foamnordic_connector_api_v1* api;
            void* owner;
            ~Release() { api->release(owner); }
        } release{&library_->api(), result.owner};
        if (result.data == nullptr || result.owner == nullptr) {
            throw std::runtime_error("Connector returned an empty result owner.");
        }
        const auto element = static_cast<bindnordic::Element>(result.element);
        const auto element_bytes = bindnordic::element_size(element);
        if (element_bytes == 0 || result.rows == 0 || result.columns == 0
            || result.rows > std::numeric_limits<std::uint64_t>::max() / result.columns
            || result.rows * result.columns
                > std::numeric_limits<std::uint64_t>::max() / element_bytes
            || result.bytes != result.rows * result.columns * element_bytes) {
            throw std::runtime_error("Connector returned an invalid tensor layout.");
        }
        std::vector<std::byte> bytes(result.bytes);
        std::memcpy(bytes.data(), result.data, bytes.size());
        return {
            "bindnordic.predictions",
            element,
            {result.rows, result.columns},
            std::move(bytes),
        };
    }

private:
    std::shared_ptr<PluginLibrary> library_;
    void* model_{nullptr};
};

class PluginConnector final : public ModelConnector {
public:
    explicit PluginConnector(std::shared_ptr<PluginLibrary> library)
        : library_(std::move(library)), id_(library_->api().id) {}

    [[nodiscard]] std::string_view id() const noexcept override { return id_; }

    [[nodiscard]] bool supports(inference::ModelFormat format) const noexcept override {
        return inference::name(format) == id_;
    }

    std::unique_ptr<inference::PackedModelKernel> load(
        std::span<const std::byte> payload,
        const inference::ModelArtifact&,
        std::uint32_t threads) const override {
        return std::make_unique<PluginKernel>(library_, payload, threads);
    }

private:
    std::shared_ptr<PluginLibrary> library_;
    std::string id_;
};

}  // namespace

void load_connector_plugin(const std::filesystem::path& path) {
    auto library = std::make_shared<PluginLibrary>(path);
    register_connector(std::make_unique<PluginConnector>(std::move(library)));
}

}  // namespace foamnordic::backend
// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * FoamNordic
 */
// clang-format on
