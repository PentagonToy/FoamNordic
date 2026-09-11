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

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
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

namespace foamnordic::inference {
namespace {

std::atomic<std::uint64_t> shared_memory_sequence{0};

fjord::FjordListener make_listener(const fjord::FjordAddress& address) {
    if (address.kind == fjord::FjordKind::unix_socket) {
        return fjord::FjordListener::local(address.location, 128);
    }
    return fjord::FjordListener::network(address.location, address.port, 128);
}

std::string json_string(std::string_view value) {
    constexpr char hexadecimal[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(value.size() + 2);
    encoded.push_back('"');
    for (const auto character : value) {
        switch (character) {
        case '"': encoded += "\\\""; break;
        case '\\': encoded += "\\\\"; break;
        case '\b': encoded += "\\b"; break;
        case '\f': encoded += "\\f"; break;
        case '\n': encoded += "\\n"; break;
        case '\r': encoded += "\\r"; break;
        case '\t': encoded += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20) {
                encoded += "\\u00";
                encoded.push_back(hexadecimal[
                    (static_cast<unsigned char>(character) >> 4) & 0x0f]);
                encoded.push_back(hexadecimal[
                    static_cast<unsigned char>(character) & 0x0f]);
            } else {
                encoded.push_back(character);
            }
        }
    }
    encoded.push_back('"');
    return encoded;
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
    ModelKernel& kernel_;
    std::mutex mutex_;
};

}  // namespace

void WorkerOptions::validate() const {
    if (connections == 0) {
        throw std::invalid_argument(
            "Native model worker connection count must be positive.");
    }
    if (model_threads == 0) {
        throw std::invalid_argument(
            "Native model worker model thread count must be positive.");
    }
    if (maximum_payload == 0) {
        throw std::invalid_argument("Native model worker payload limit must be positive.");
    }
    if (maximum_receive_bytes == 0) {
        throw std::invalid_argument(
            "Native model worker receive-memory limit must be positive.");
    }
    if (maximum_working_bytes == 0) {
        throw std::invalid_argument(
            "Native model worker working-memory limit must be positive.");
    }
    if (ucx && ucx_host.empty()) {
        throw std::invalid_argument(
            "Native model worker UCX mode requires an advertised host.");
    }
#ifndef FOAMNORDIC_HAVE_UCX
    if (ucx) {
        throw std::invalid_argument(
            "Native model worker was built without UCX support.");
    }
#endif
}

ModelWorker::ModelWorker(
    fjord::FjordAddress address,
    ModelArtifact artifact,
    const CellEvaluationPolicy& bypass,
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

ModelWorker::ModelWorker(
    fjord::FjordAddress address,
    const std::filesystem::path& manifest_path,
    const CellEvaluationPolicy& bypass,
    WorkerOptions options)
    : requested_address_(std::move(address)),
      listener_(make_listener(requested_address_)),
      bypass_(bypass),
      options_(options) {
    requested_address_.validate();
    options_.validate();
    auto capacity = std::make_shared<ExecutionCapacityPool>(
        options_.model_threads);
    auto loaded = load_model(
        manifest_path,
        {
            options_.model_threads,
            std::min(options_.connections, options_.model_threads),
            options_.maximum_working_bytes,
            options_.profile_every,
            std::move(capacity),
            options_.model_threads,
        });
    artifact_ = std::move(loaded.artifact);
    owned_kernel_ = std::move(loaded.kernel);
    kernel_ = owned_kernel_.get();
}

fjord::FjordAddress ModelWorker::address() const {
    return listener_.address();
}

void ModelWorker::run() {
    native::log(
        native::LogLevel::info,
        "Model worker listening at " + listener_.address().text());
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
                "Native model worker UCX mode requires a TCP control address.");
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
                "Model worker required UCX but the solver did not negotiate it.");
        }
        if (!ranks.insert(session.rank).second) {
            throw std::runtime_error(
                "Model worker received a duplicate solver rank.");
        }
        if (connection == 0) {
            session_id = session.session_id;
            global_peers = session.peers;
        } else if (session.session_id != session_id
                   || session.peers != global_peers) {
            throw std::runtime_error(
                "Model worker sessions disagree on MPI identity.");
        }
        if (fjord::any(session.capabilities & fjord::Capability::ucx)) {
#ifdef FOAMNORDIC_HAVE_UCX
            auto ucx_listener = fjord::UcxListener::network(options_.ucx_host, 0);
            harbor->offer_ucx(ucx_listener);
            native::log(
                native::LogLevel::info,
                "Model worker rank " + std::to_string(session.rank)
                    + " data plane: UCX");
#else
            throw std::runtime_error(
                "Model worker selected UCX without UCX build support.");
#endif
        } else if (fjord::any(session.capabilities & fjord::Capability::shm)) {
            const auto shared_memory_name =
                std::string("/fnc-")
                + std::to_string(static_cast<long long>(::getpid())) + '-'
                + std::to_string(shared_memory_sequence.fetch_add(
                    1, std::memory_order_relaxed)) + '-'
                + std::to_string(session.session_id) + '-'
                + std::to_string(session.rank);
            harbor->offer_shared_memory(shared_memory_name);
            native::log(
                native::LogLevel::info,
                "Model worker rank " + std::to_string(session.rank)
                    + " data plane: SHM");
        } else if (requested_address_.kind == fjord::FjordKind::unix_socket) {
            native::log(
                native::LogLevel::info,
                "Model worker rank " + std::to_string(session.rank)
                    + " data plane: UDS");
        } else {
            native::log(
                native::LogLevel::info,
                "Model worker rank " + std::to_string(session.rank)
                    + " data plane: TCP");
        }
        harbors.push_back(std::move(harbor));
    }

    SynchronizedKernel synchronized(*kernel_);
    ModelKernel& execution_kernel = kernel_->owns_backend_synchronization()
                                       ? *kernel_
                                       : static_cast<ModelKernel&>(synchronized);
    ReceiveCreditPool receive_credits(options_.maximum_receive_bytes);
    ReceiveCreditPool output_credits(options_.maximum_working_bytes);
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::vector<std::thread> runners;
    runners.reserve(harbors.size());
    try {
        for (auto& harbor : harbors) {
            runners.emplace_back([&, peer = harbor.get()] {
                try {
                    InferenceRunner runner(
                        *peer,
                        artifact_.contract,
                        bypass_,
                        execution_kernel,
                        options_.maximum_payload,
                        &receive_credits,
                        &output_credits);
                    runner.run();
                } catch (...) {
                    {
                        std::scoped_lock lock(failure_mutex);
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                    execution_kernel.cancel_pending();
                    receive_credits.cancel();
                    output_credits.cancel();
                    for (auto& active : harbors) {
                        active->interrupt();
                    }
                }
            });
        }
    } catch (...) {
        execution_kernel.cancel_pending();
        receive_credits.cancel();
        output_credits.cancel();
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
    listener_.close();
    if (execution_kernel.telemetry_enabled()) {
        const auto receive = receive_credits.snapshot();
        const auto output = output_credits.snapshot();
        const auto resources = execution_kernel.execution_snapshot();
        const auto samples = resources.profile_samples;
        const auto mean_microseconds = [samples](std::uint64_t nanoseconds) {
            return samples == 0 ? 0 : nanoseconds / samples / 1000;
        };
        native::log(
            native::LogLevel::info,
            "Model telemetry: {\"program\":"
                + json_string(
                    options_.program.empty() ? artifact_.contract.name : options_.program)
                + ",\"connections\":" + std::to_string(options_.connections)
                + ",\"model_threads\":" + std::to_string(options_.model_threads)
                + ",\"requests\":" + std::to_string(resources.requests)
                + ",\"completed_exchanges\":"
                + std::to_string(resources.completed_exchanges)
                + ",\"rows\":" + std::to_string(resources.rows)
                + ",\"active_cells\":" + std::to_string(resources.active_cells)
                + ",\"input_payload_bytes\":"
                + std::to_string(resources.input_payload_bytes)
                + ",\"output_payload_bytes\":"
                + std::to_string(resources.output_payload_bytes)
                + ",\"peak_rows\":" + std::to_string(resources.peak_rows)
                + ",\"peak_active_cells\":"
                + std::to_string(resources.peak_active_cells)
                + ",\"peak_input_payload_bytes\":"
                + std::to_string(resources.peak_input_payload_bytes)
                + ",\"peak_output_payload_bytes\":"
                + std::to_string(resources.peak_output_payload_bytes)
                + ",\"peak_active_contexts\":"
                + std::to_string(resources.peak_active_contexts)
                + ",\"peak_waiting_contexts\":"
                + std::to_string(resources.peak_waiting_contexts)
                + ",\"peak_working_bytes\":"
                + std::to_string(resources.peak_admitted_bytes)
                + ",\"retained_workspace_bytes\":"
                + std::to_string(resources.retained_workspace_bytes)
                + ",\"receive_reservations\":" + std::to_string(receive.requests)
                + ",\"peak_receive_bytes\":"
                + std::to_string(receive.peak_active_bytes)
                + ",\"publication_reservations\":"
                + std::to_string(output.requests)
                + ",\"peak_publication_bytes\":"
                + std::to_string(output.peak_active_bytes)
                + ",\"profile_samples\":" + std::to_string(samples)
                + ",\"mean_admission_us\":"
                + std::to_string(mean_microseconds(resources.admission_nanoseconds))
                + ",\"mean_backend_wait_us\":"
                + std::to_string(mean_microseconds(resources.backend_wait_nanoseconds))
                + ",\"mean_pack_us\":"
                + std::to_string(mean_microseconds(resources.packing_nanoseconds))
                + ",\"mean_infer_us\":"
                + std::to_string(mean_microseconds(resources.inference_nanoseconds))
                + ",\"mean_output_us\":"
                + std::to_string(mean_microseconds(resources.output_nanoseconds))
                + "}");
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}  // namespace foamnordic::inference
