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

#include "foamnordic/backend/inference/engine.hpp"

#include "foamnordic/backend/inference/residency.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace foamnordic::inference {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_nanoseconds(Clock::time_point start) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
            .count());
}

std::uint64_t tensor_bytes(const TensorMap& tensors) {
    std::uint64_t total = 0;
    for (const auto& [name, tensor] : tensors) {
        (void)name;
        const auto bytes = static_cast<std::uint64_t>(tensor.bytes.size());
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error("Stateful engine tensor bytes overflowed.");
        }
        total += bytes;
    }
    return total;
}

std::uint64_t tensor_rows(const TensorMap& tensors) noexcept {
    std::uint64_t rows = 0;
    for (const auto& [name, tensor] : tensors) {
        (void)name;
        if (!tensor.shape.empty()) {
            rows = std::max(rows, tensor.shape.front());
        }
    }
    return rows;
}

}  // namespace

void EngineStep::validate() const {
    if (!std::isfinite(physical_time) || !std::isfinite(delta_time)
        || delta_time <= 0.0) {
        throw std::invalid_argument("Stateful engine step has invalid time metadata.");
    }
}

std::vector<std::byte> StatefulEngineBackend::checkpoint(
    const TensorMap&) const {
    throw std::logic_error("Stateful engine does not support checkpoints.");
}

TensorMap StatefulEngineBackend::restore(std::span<const std::byte>) {
    throw std::logic_error("Stateful engine does not support restore.");
}

TransactionalEngine::TransactionalEngine(
    ProgramContract contract,
    ExecutionResources resources,
    StatefulEngineBackend& backend,
    EngineCostObserver* observer)
    : contract_(std::move(contract)),
      resources_(std::move(resources)),
      backend_(backend),
      engine_capabilities_(backend.engine_capabilities()),
      backend_capabilities_(backend.backend_capabilities()),
      observer_(observer) {
    contract_.validate();
    resources_.validate();
    const auto residency = plan_residency(backend_capabilities_);
    if (!backend_capabilities_.valid()) {
        throw std::invalid_argument("Stateful engine memory capabilities are invalid.");
    }
    if (!residency.direct()) {
        throw std::invalid_argument(
            "Stateful engine requires synchronous host tensor boundaries for this execution path.");
    }
    if (engine_capabilities_.restore && !engine_capabilities_.checkpoint) {
        throw std::invalid_argument(
            "Stateful engine restore requires checkpoint support.");
    }
}

TransactionalEngine::~TransactionalEngine() { close(); }

void TransactionalEngine::initialise(
    TensorMap state,
    std::uint64_t topology_generation) {
    if (active_) {
        throw std::logic_error("Stateful engine is already initialised.");
    }
    validate_tensors(state, "initial state");
    backend_.initialise(contract_, resources_);
    state_ = std::move(state);
    topology_generation_ = topology_generation;
    state_generation_ = 0;
    active_ = true;
}

TensorMap TransactionalEngine::advance(
    const TensorMap& inputs,
    const EngineStep& step) {
    if (!active_) {
        throw std::logic_error("Stateful engine is not initialised.");
    }
    step.validate();
    validate_tensors(inputs, "input");
    if (step.topology_generation != topology_generation_
        && !engine_capabilities_.survives_topology_change) {
        throw std::runtime_error(
            "Stateful engine requires reinitialisation for the topology generation.");
    }
    const auto started = Clock::now();
    EngineCostObservation observation{
        .operation = EngineOperation::advance,
        .outcome = EngineOutcome::failed,
        .exchange_index = step.exchange_index,
        .state_generation = state_generation_,
        .topology_generation = step.topology_generation,
        .rows = tensor_rows(inputs),
        .input_bytes = tensor_bytes(inputs),
        .state_bytes = tensor_bytes(state_),
    };
    try {
        auto candidate = backend_.advance(state_, inputs, step);
        observation.transfer_nanoseconds = candidate.transfer_nanoseconds;
        observation.queue_nanoseconds = candidate.queue_nanoseconds;
        observation.compute_nanoseconds = candidate.compute_nanoseconds;
        observation.device_memory_bytes = candidate.device_memory_bytes;
        observation.retries = candidate.retries;
        observation.fallbacks = candidate.fallbacks;
        validate_tensors(candidate.state, "candidate state");
        validate_tensors(candidate.outputs, "output");
        observation.output_bytes = tensor_bytes(candidate.outputs);
        const auto commit_started = Clock::now();
        state_ = std::move(candidate.state);
        topology_generation_ = step.topology_generation;
        ++state_generation_;
        observation.commit_nanoseconds = elapsed_nanoseconds(commit_started);
        observation.state_generation = state_generation_;
        observation.state_bytes = tensor_bytes(state_);
        observation.outcome = EngineOutcome::completed;
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
        return std::move(candidate.outputs);
    } catch (...) {
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
        throw;
    }
}

EngineCheckpoint TransactionalEngine::checkpoint() const {
    if (!active_) {
        throw std::logic_error("Stateful engine is not initialised.");
    }
    if (!engine_capabilities_.checkpoint) {
        throw std::logic_error("Stateful engine does not support checkpoints.");
    }
    const auto started = Clock::now();
    EngineCostObservation observation{
        .operation = EngineOperation::checkpoint,
        .outcome = EngineOutcome::failed,
        .state_generation = state_generation_,
        .topology_generation = topology_generation_,
        .state_bytes = tensor_bytes(state_),
    };
    try {
        auto payload = backend_.checkpoint(state_);
        if (payload.empty()) {
            throw std::runtime_error("Stateful engine produced an empty checkpoint.");
        }
        observation.output_bytes = payload.size();
        observation.outcome = EngineOutcome::completed;
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
        return {state_generation_, topology_generation_, std::move(payload)};
    } catch (...) {
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
        throw;
    }
}

void TransactionalEngine::restore(const EngineCheckpoint& checkpoint) {
    if (!active_) {
        throw std::logic_error("Stateful engine is not initialised.");
    }
    if (!engine_capabilities_.restore) {
        throw std::logic_error("Stateful engine does not support restore.");
    }
    if (checkpoint.payload.empty()) {
        throw std::invalid_argument("Stateful engine checkpoint is empty.");
    }
    const auto started = Clock::now();
    EngineCostObservation observation{
        .operation = EngineOperation::restore,
        .outcome = EngineOutcome::failed,
        .state_generation = state_generation_,
        .topology_generation = checkpoint.topology_generation,
        .input_bytes = checkpoint.payload.size(),
        .state_bytes = tensor_bytes(state_),
    };
    try {
        auto candidate = backend_.restore(checkpoint.payload);
        validate_tensors(candidate, "restored state");
        const auto commit_started = Clock::now();
        state_ = std::move(candidate);
        state_generation_ = checkpoint.state_generation;
        topology_generation_ = checkpoint.topology_generation;
        observation.commit_nanoseconds = elapsed_nanoseconds(commit_started);
        observation.state_generation = state_generation_;
        observation.state_bytes = tensor_bytes(state_);
        observation.outcome = EngineOutcome::completed;
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
    } catch (...) {
        observation.elapsed_nanoseconds = elapsed_nanoseconds(started);
        if (observer_ != nullptr) {
            observer_->observe(observation);
        }
        throw;
    }
}

void TransactionalEngine::close() noexcept {
    if (active_) {
        backend_.close();
        state_.clear();
        active_ = false;
    }
}

const TensorMap& TransactionalEngine::state() const noexcept { return state_; }

std::uint64_t TransactionalEngine::state_generation() const noexcept {
    return state_generation_;
}

std::uint64_t TransactionalEngine::topology_generation() const noexcept {
    return topology_generation_;
}

bool TransactionalEngine::active() const noexcept { return active_; }

void TransactionalEngine::validate_tensors(
    const TensorMap& tensors,
    const char* kind) {
    for (const auto& [name, tensor] : tensors) {
        if (name.empty() || name != tensor.name) {
            throw std::invalid_argument(
                std::string("Stateful engine ") + kind
                + " tensor key does not match its name.");
        }
        tensor.view().validate();
    }
}

}  // namespace foamnordic::inference
