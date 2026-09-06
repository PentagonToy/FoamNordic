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

#include <exception>
#include <stdexcept>
#include <utility>

#include "foamnordic/runtime/log.hpp"

namespace foamnordic::inference {

InferenceRunner::InferenceRunner(
    fjord::Harbor& harbor,
    ProgramContract contract,
    const CellEvaluationPolicy& bypass,
    ModelKernel& kernel,
    std::uint64_t maximum_exchange_payload,
    ReceiveCreditPool* receive_credits)
    : harbor_(harbor),
      exchange_(std::move(contract)),
      bypass_(bypass),
      kernel_(kernel),
      maximum_exchange_payload_(maximum_exchange_payload),
      receive_credits_(receive_credits) {
    if (maximum_exchange_payload_ == 0) {
        throw std::invalid_argument(
            "Model exchange payload limit must be positive.");
    }
}

bool InferenceRunner::run_one() {
    std::optional<ReceiveCreditPool::Reservation> receive_reservation;
    if (receive_credits_ != nullptr) {
        receive_reservation.emplace(*receive_credits_);
    }
    bool started = false;
    std::uint32_t received_tensors = 0;
    std::uint64_t received_payload_bytes = 0;
    try {
        while (true) {
            auto message = harbor_.receive_message(
                maximum_exchange_payload_ - received_payload_bytes,
                receive_reservation.has_value()
                    ? fjord::PayloadAdmission{
                          &*receive_reservation,
                          [](void* context, std::uint64_t bytes) {
                              static_cast<ReceiveCreditPool::Reservation*>(context)
                                  ->acquire(bytes);
                          }}
                    : fjord::PayloadAdmission{});
            if (message.kind == fjord::RuneKind::shutdown) {
                if (started) {
                    throw std::runtime_error("Model exchange was interrupted by shutdown.");
                }
                return false;
            }
            if (message.kind == fjord::RuneKind::error) {
                throw std::runtime_error("The remote endpoint reported an exchange error.");
            }
            if (message.kind == fjord::RuneKind::tensor) {
                if (!message.tensor.has_value() || message.tensor->shape.empty()) {
                    throw std::runtime_error("Model input tensor is incomplete.");
                }
                if (!started) {
                    exchange_.begin(
                        message.tensor->time_index,
                        message.tensor->physical_time,
                        message.tensor->shape.front(),
                        message.tensor->solver_time_index);
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
                if (message.tensor_count != received_tensors) {
                    throw std::runtime_error(
                        "Model atomic commit does not match its prepared tensor batch.");
                }
                break;
            }
            throw std::runtime_error("Model runner received an unsupported message.");
        }

        exchange_.seal_inputs();
        if (receive_reservation) {
            receive_reservation->seal();
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
        return true;
    } catch (...) {
        if (receive_credits_) {
            receive_credits_->cancel();
        }
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
