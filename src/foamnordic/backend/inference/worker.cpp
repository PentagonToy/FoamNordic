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

#include "foamnordic/backend/inference/worker.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

#include "foamnordic/backend/inference/load.hpp"
#include "foamnordic/runtime/log.hpp"

namespace foamnordic::closure {
namespace {

fjord::FjordListener make_listener(const fjord::FjordAddress& address) {
    if (address.kind == fjord::FjordKind::unix_socket) {
        return fjord::FjordListener::local(address.location);
    }
    return fjord::FjordListener::network(address.location, address.port);
}

}  // namespace

void WorkerOptions::validate() const {
    if (maximum_payload == 0) {
        throw std::invalid_argument("Native closure worker payload limit must be positive.");
    }
}

NativeClosureWorker::NativeClosureWorker(
    fjord::FjordAddress address,
    ModelArtifact artifact,
    const BypassPolicy& bypass,
    ModelKernel& kernel,
    WorkerOptions options)
    : requested_address_(std::move(address)),
      listener_(make_listener(requested_address_)),
      artifact_(std::move(artifact)),
      bypass_(bypass),
      kernel_(&kernel),
      options_(options) {
    requested_address_.validate();
    artifact_.validate();
    options_.validate();
}

NativeClosureWorker::NativeClosureWorker(
    fjord::FjordAddress address,
    const std::filesystem::path& manifest_path,
    const BypassPolicy& bypass,
    WorkerOptions options)
    : requested_address_(std::move(address)),
      listener_(make_listener(requested_address_)),
      bypass_(bypass),
      options_(options) {
    requested_address_.validate();
    options_.validate();
    auto loaded = load_model(manifest_path);
    artifact_ = std::move(loaded.artifact);
    owned_kernel_ = std::move(loaded.kernel);
    kernel_ = owned_kernel_.get();
}

fjord::FjordAddress NativeClosureWorker::address() const {
    return listener_.address();
}

void NativeClosureWorker::run() {
    native::log(
        native::LogLevel::info,
        "Closure worker listening at " + listener_.address().text());
    fjord::Harbor harbor(listener_.accept());
    auto capabilities = requested_address_.kind == fjord::FjordKind::unix_socket
                            ? fjord::Capability::uds
                            : fjord::Capability::tcp;
    if (options_.shared_memory
        && requested_address_.kind == fjord::FjordKind::unix_socket) {
        capabilities = capabilities | fjord::Capability::shm;
    }
    const auto session = harbor.accept_session({
        1,
        capabilities,
        0,
        1,
        options_.maximum_payload,
    });
    if (fjord::any(session.capabilities & fjord::Capability::shm)) {
        const auto shared_memory_name =
            std::string("/foamnordic-closure-")
            + std::to_string(static_cast<long long>(::getpid())) + '-'
            + std::to_string(session.session_id);
        harbor.offer_shared_memory(shared_memory_name);
        native::log(native::LogLevel::info, "Closure worker data plane: SHM");
    } else if (requested_address_.kind == fjord::FjordKind::unix_socket) {
        native::log(native::LogLevel::info, "Closure worker data plane: UDS");
    } else {
        native::log(native::LogLevel::info, "Closure worker data plane: TCP");
    }
    NativeClosureRunner runner(
        harbor,
        artifact_.contract,
        bypass_,
        *kernel_);
    runner.run();
}

}  // namespace foamnordic::closure
