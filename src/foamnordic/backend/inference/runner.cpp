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

namespace foamnordic::closure {

NativeClosureRunner::NativeClosureRunner(
    fjord::Harbor& harbor,
    ClosureContract contract,
    const BypassPolicy& bypass,
    ModelKernel& kernel)
    : harbor_(harbor), exchange_(std::move(contract)), bypass_(bypass), kernel_(kernel) {}

bool NativeClosureRunner::run_one() {
    bool started = false;
    std::uint32_t received_tensors = 0;
    try {
        while (true) {
            auto message = harbor_.receive_message();
            if (message.kind == fjord::RuneKind::shutdown) {
                if (started) {
                    throw std::runtime_error("Closure exchange was interrupted by shutdown.");
                }
                return false;
            }
            if (message.kind == fjord::RuneKind::error) {
                throw std::runtime_error("The remote endpoint reported an exchange error.");
            }
            if (message.kind == fjord::RuneKind::tensor) {
                if (!message.tensor.has_value() || message.tensor->shape.empty()) {
                    throw std::runtime_error("Closure input tensor is incomplete.");
                }
                if (!started) {
                    exchange_.begin(
                        message.tensor->time_index,
                        message.tensor->physical_time,
                        message.tensor->shape.front(),
                        message.tensor->solver_time_index);
                    started = true;
                }
                exchange_.add_input(std::move(*message.tensor));
                ++received_tensors;
                continue;
            }
            if (message.kind == fjord::RuneKind::complete) {
                if (!started || message.exchange_index != exchange_.exchange_index()) {
                    throw std::runtime_error("Closure completion marker does not match its batch.");
                }
                if (message.tensor_count != received_tensors) {
                    throw std::runtime_error(
                        "Closure atomic commit does not match its prepared tensor batch.");
                }
                break;
            }
            throw std::runtime_error("Closure runner received an unsupported message.");
        }

        exchange_.seal_inputs();
        const auto& active_cells = exchange_.prepare(bypass_);
        auto predictions = kernel_.evaluate(
            exchange_.inputs(),
            active_cells,
            exchange_.exchange_index(),
            exchange_.physical_time());
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
    } catch (const std::exception& error) {
        exchange_.fail(error.what());
        try {
            harbor_.fail_exchange(
                started ? exchange_.exchange_index() : 0);
        } catch (...) {
        }
        throw;
    }
}

void NativeClosureRunner::run() {
    native::log(native::LogLevel::info, "Native closure runner started.");
    while (run_one()) {
    }
    native::log(native::LogLevel::info, "Native closure runner stopped.");
}

}  // namespace foamnordic::closure
