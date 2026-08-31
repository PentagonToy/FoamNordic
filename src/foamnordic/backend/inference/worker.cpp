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

#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include "foamnordic/backend/inference/load.hpp"
#include "foamnordic/runtime/log.hpp"
#ifdef FOAMNORDIC_HAVE_UCX
#include "foamnordic/fjord/ucx_channel.hpp"
#endif

namespace foamnordic::closure {
namespace {

fjord::FjordListener make_listener(const fjord::FjordAddress& address) {
    if (address.kind == fjord::FjordKind::unix_socket) {
        return fjord::FjordListener::local(address.location, 128);
    }
    return fjord::FjordListener::network(address.location, address.port, 128);
}

class SynchronizedKernel final : public ModelKernel {
public:
    explicit SynchronizedKernel(ModelKernel& kernel) : kernel_(kernel) {}

    TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank) override {
        std::scoped_lock lock(mutex_);
        return kernel_.evaluate(
            inputs, active_cells, exchange_index, physical_time, rank);
    }

private:
    ModelKernel& kernel_;
    std::mutex mutex_;
};

}  // namespace

void WorkerOptions::validate() const {
    if (connections == 0) {
        throw std::invalid_argument(
            "Native closure worker connection count must be positive.");
    }
    if (model_threads == 0) {
        throw std::invalid_argument(
            "Native closure worker model thread count must be positive.");
    }
    if (maximum_payload == 0) {
        throw std::invalid_argument("Native closure worker payload limit must be positive.");
    }
    if (ucx && ucx_host.empty()) {
        throw std::invalid_argument(
            "Native closure worker UCX mode requires an advertised host.");
    }
#ifndef FOAMNORDIC_HAVE_UCX
    if (ucx) {
        throw std::invalid_argument(
            "Native closure worker was built without UCX support.");
    }
#endif
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
    auto loaded = load_model(manifest_path, {options_.model_threads});
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
    auto capabilities = requested_address_.kind == fjord::FjordKind::unix_socket
                            ? fjord::Capability::uds
                            : fjord::Capability::tcp;
    if (options_.shared_memory
        && requested_address_.kind == fjord::FjordKind::unix_socket) {
        capabilities = capabilities | fjord::Capability::shm;
    }
    if (options_.ucx) {
        if (requested_address_.kind != fjord::FjordKind::tcp) {
            throw std::invalid_argument(
                "Native closure worker UCX mode requires a TCP control address.");
        }
        capabilities = capabilities | fjord::Capability::ucx;
    }
    std::vector<std::unique_ptr<fjord::Harbor>> harbors;
    harbors.reserve(options_.connections);
    std::unordered_set<std::uint32_t> ranks;
    std::uint64_t session_id = 0;
    std::uint32_t global_peers = 0;
    for (std::uint32_t connection = 0;
         connection < options_.connections;
         ++connection) {
        auto harbor = std::make_unique<fjord::Harbor>(listener_.accept());
        const auto session = harbor->accept_session({
            1,
            capabilities,
            0,
            1,
            options_.maximum_payload,
        });
        if (options_.ucx
            && !fjord::any(session.capabilities & fjord::Capability::ucx)) {
            throw std::runtime_error(
                "Closure worker required UCX but the solver did not negotiate it.");
        }
        if (!ranks.insert(session.rank).second) {
            throw std::runtime_error(
                "Closure worker received a duplicate solver rank.");
        }
        if (connection == 0) {
            session_id = session.session_id;
            global_peers = session.peers;
        } else if (session.session_id != session_id
                   || session.peers != global_peers) {
            throw std::runtime_error(
                "Closure worker sessions disagree on MPI identity.");
        }
        if (fjord::any(session.capabilities & fjord::Capability::ucx)) {
#ifdef FOAMNORDIC_HAVE_UCX
            auto ucx_listener = fjord::UcxListener::network(options_.ucx_host, 0);
            harbor->offer_ucx(ucx_listener);
            native::log(
                native::LogLevel::info,
                "Closure worker rank " + std::to_string(session.rank)
                    + " data plane: UCX");
#else
            throw std::runtime_error(
                "Closure worker selected UCX without UCX build support.");
#endif
        } else if (fjord::any(session.capabilities & fjord::Capability::shm)) {
            const auto shared_memory_name =
                std::string("/fnc-")
                + std::to_string(static_cast<long long>(::getpid())) + '-'
                + std::to_string(session.session_id) + '-'
                + std::to_string(session.rank);
            harbor->offer_shared_memory(shared_memory_name);
            native::log(
                native::LogLevel::info,
                "Closure worker rank " + std::to_string(session.rank)
                    + " data plane: SHM");
        } else if (requested_address_.kind == fjord::FjordKind::unix_socket) {
            native::log(
                native::LogLevel::info,
                "Closure worker rank " + std::to_string(session.rank)
                    + " data plane: UDS");
        } else {
            native::log(
                native::LogLevel::info,
                "Closure worker rank " + std::to_string(session.rank)
                    + " data plane: TCP");
        }
        harbors.push_back(std::move(harbor));
    }

    SynchronizedKernel synchronized(*kernel_);
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::vector<std::thread> runners;
    runners.reserve(harbors.size());
    try {
        for (auto& harbor : harbors) {
            runners.emplace_back([&, peer = harbor.get()] {
                try {
                    NativeClosureRunner runner(
                        *peer,
                        artifact_.contract,
                        bypass_,
                        synchronized);
                    runner.run();
                } catch (...) {
                    {
                        std::scoped_lock lock(failure_mutex);
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                    for (auto& active : harbors) {
                        active->interrupt();
                    }
                }
            });
        }
    } catch (...) {
        for (auto& active : harbors) {
            active->interrupt();
        }
        for (auto& runner : runners) {
            runner.join();
        }
        throw;
    }
    for (auto& runner : runners) {
        runner.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}  // namespace foamnordic::closure
