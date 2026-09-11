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

#include "foamnordic/backend/inference/runner.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <limits>
#include <utility>

#include "foamnordic/runtime/log.hpp"

namespace foamnordic::inference {

InferenceRunner::InferenceRunner(
    fjord::Harbor& harbor,
    ProgramContract contract,
    const CellEvaluationPolicy& bypass,
    ModelKernel& kernel,
    std::uint64_t maximum_exchange_payload,
    ReceiveCreditPool* receive_credits,
    ReceiveCreditPool* output_credits)
    : harbor_(harbor),
      exchange_(contract),
      bypass_(bypass),
      kernel_(kernel),
      maximum_exchange_payload_(maximum_exchange_payload),
      receive_credits_(receive_credits),
      output_credits_(output_credits),
      telemetry_enabled_(kernel.telemetry_enabled()) {
    for (const auto& field : contract.outputs) {
        const auto width = fjord::element_size(field.element);
        if (field.components > (std::numeric_limits<std::uint64_t>::max() - output_bytes_per_cell_) / width)
            throw std::overflow_error("Model output contract byte count overflowed.");
        output_bytes_per_cell_ += field.components * width;
    }
    if (maximum_exchange_payload_ == 0) {
        throw std::invalid_argument(
            "Model exchange payload limit must be positive.");
    }
}

bool InferenceRunner::run_one() {
    std::optional<ReceiveCreditPool::Reservation> receive_reservation;
    std::optional<ReceiveCreditPool::Reservation> output_reservation;
    if (receive_credits_ != nullptr) {
        receive_reservation.emplace(*receive_credits_);
    }
    bool started = false;
    bool announced = false;
    std::uint32_t announced_tensors = 0;
    std::uint64_t announced_payload_bytes = 0;
    std::uint64_t announced_topology_generation = 0;
    std::uint32_t received_tensors = 0;
    std::uint64_t received_payload_bytes = 0;
    try {
        while (true) {
            const auto remaining_payload = maximum_exchange_payload_ - received_payload_bytes;
            const auto admitted_remaining = announced
                ? announced_payload_bytes - received_payload_bytes
                : remaining_payload;
            auto message = harbor_.receive_message(
                std::min(remaining_payload, admitted_remaining),
                receive_reservation.has_value() && !announced
                    ? fjord::PayloadAdmission{
                          &*receive_reservation,
                          [](void* context, std::uint64_t bytes) {
                              static_cast<ReceiveCreditPool::Reservation*>(context)
                                  ->acquire(bytes);
                          }}
                    : fjord::PayloadAdmission{},
                true);
            if (message.kind == fjord::RuneKind::shutdown) {
                if (started) {
                    throw std::runtime_error("Model exchange was interrupted by shutdown.");
                }
                return false;
            }
            if (message.kind == fjord::RuneKind::error) {
                throw std::runtime_error("The remote endpoint reported an exchange error.");
            }
            if (message.kind == fjord::RuneKind::request) {
                if (started || announced) {
                    throw std::runtime_error("Model request metadata arrived out of order.");
                }
                if (message.payload_bytes > maximum_exchange_payload_) {
                    throw std::runtime_error("Model request exceeds the exchange payload limit.");
                }
                announced = true;
                announced_tensors = message.tensor_count;
                announced_payload_bytes = message.payload_bytes;
                announced_topology_generation = message.topology_generation;
                if (receive_reservation) {
                    receive_reservation->acquire(announced_payload_bytes);
                    receive_reservation->seal();
                }
                continue;
            }
            if (message.kind == fjord::RuneKind::tensor) {
                if (!message.tensor.has_value() || message.tensor->shape.empty()) {
                    throw std::runtime_error("Model input tensor is incomplete.");
                }
                if (!started) {
                    if (announced
                        && message.tensor->topology_generation
                               != announced_topology_generation) {
                        throw std::runtime_error(
                            "Model request topology does not match its admission metadata.");
                    }
                    exchange_.begin(
                        message.tensor->time_index,
                        message.tensor->physical_time,
                        message.tensor->shape.front(),
                        message.tensor->solver_time_index,
                        message.tensor->topology_generation);
                    started = true;
                }
                received_payload_bytes += message.tensor->bytes.size();
                exchange_.add_input(std::move(*message.tensor));
                ++received_tensors;
                continue;
            }
            if (message.kind == fjord::RuneKind::complete) {
                if (!started || message.exchange_index != exchange_.exchange_index()) {
                    throw std::runtime_error("Model completion marker does not match its batch.");
                }
                if (message.topology_generation
                    != exchange_.topology_generation()) {
                    throw std::runtime_error(
                        "Model completion marker does not match its topology generation.");
                }
                if (message.tensor_count != received_tensors) {
                    throw std::runtime_error(
                        "Model atomic commit does not match its prepared tensor batch.");
                }
                if (announced && (announced_tensors != received_tensors
                    || announced_payload_bytes != received_payload_bytes)) {
                    throw std::runtime_error(
                        "Model request body does not match its admission metadata.");
                }
                break;
            }
            throw std::runtime_error("Model runner received an unsupported message.");
        }

        exchange_.seal_inputs();
        if (receive_reservation) {
            receive_reservation->seal();
        }
        if (output_credits_) {
            const auto cells = exchange_.cell_count();
            if (output_bytes_per_cell_ > std::numeric_limits<std::uint64_t>::max() / 2
                || (output_bytes_per_cell_ && cells > std::numeric_limits<std::uint64_t>::max() / 2 / output_bytes_per_cell_))
                throw std::overflow_error("Model publication byte count overflowed.");
            output_reservation.emplace(*output_credits_);
            // Sparse predictions and full-field bypass outputs may coexist.
            output_reservation->acquire(2 * cells * output_bytes_per_cell_);
            output_reservation->seal();
        }
        const auto& active_cells = exchange_.prepare(bypass_);
        auto predictions = kernel_.evaluate(
            exchange_.inputs(),
            active_cells,
            exchange_.exchange_index(),
            exchange_.physical_time(),
            harbor_.rank());
        for (auto& [name, tensor] : predictions) {
            (void)name;
            tensor.solver_time_index = exchange_.solver_time_index();
            tensor.topology_generation = exchange_.topology_generation();
            exchange_.add_output(std::move(tensor));
        }
        exchange_.seal_outputs(bypass_);
        auto outputs = exchange_.finish();
        std::vector<fjord::TensorView> output_views;
        output_views.reserve(outputs.size());
        for (const auto& [name, tensor] : outputs) {
            (void)name;
            output_views.push_back(tensor.view());
        }
        harbor_.publish(exchange_.exchange_index(), output_views);
        if (telemetry_enabled_) {
            std::uint64_t output_payload_bytes = 0;
            for (const auto& output : output_views) {
                if (output.bytes.size()
                    > std::numeric_limits<std::uint64_t>::max()
                        - output_payload_bytes) {
                    throw std::overflow_error(
                        "Model output payload byte count overflowed.");
                }
                output_payload_bytes += output.bytes.size();
            }
            kernel_.observe_exchange(
                exchange_.cell_count(),
                active_cells.size(),
                received_payload_bytes,
                output_payload_bytes);
        }
        return true;
    } catch (...) {
        if (receive_credits_) {
            receive_credits_->cancel();
        }
        if (output_credits_) output_credits_->cancel();
        std::string reason = "Unknown model exception.";
        try {
            throw;
        } catch (const std::exception& error) {
            reason = error.what();
        } catch (...) {
        }
        exchange_.fail(std::move(reason));
        try {
            harbor_.fail_exchange(
                started ? exchange_.exchange_index() : 0);
        } catch (...) {
        }
        throw;
    }
}

void InferenceRunner::run() {
    native::log(native::LogLevel::info, "Native model runner started.");
    while (run_one()) {
    }
    native::log(native::LogLevel::info, "Native model runner stopped.");
}

}  // namespace foamnordic::inference
