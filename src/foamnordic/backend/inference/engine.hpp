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

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "bindnordic/kernel.hpp"
#include "foamnordic/backend/inference/exchange.hpp"
#include "foamnordic/backend/inference/execution.hpp"

namespace foamnordic::inference {

struct StatefulEngineCapabilities {
    bool checkpoint{false};
    bool restore{false};
    bool survives_topology_change{false};
};

struct EngineStep {
    std::uint64_t exchange_index{0};
    std::uint64_t solver_time_index{0};
    std::uint64_t topology_generation{0};
    double physical_time{0.0};
    double delta_time{0.0};

    void validate() const;
};

struct EngineAdvance {
    TensorMap state;
    TensorMap outputs;
    std::uint64_t transfer_nanoseconds{0};
    std::uint64_t queue_nanoseconds{0};
    std::uint64_t compute_nanoseconds{0};
    std::uint64_t device_memory_bytes{0};
    std::uint64_t retries{0};
    std::uint64_t fallbacks{0};
};

enum class EngineOperation : std::uint8_t {
    advance = 0,
    checkpoint = 1,
    restore = 2,
};

enum class EngineOutcome : std::uint8_t {
    completed = 0,
    failed = 1,
};

struct EngineCostObservation {
    EngineOperation operation{EngineOperation::advance};
    EngineOutcome outcome{EngineOutcome::completed};
    std::uint64_t exchange_index{0};
    std::uint64_t state_generation{0};
    std::uint64_t topology_generation{0};
    std::uint64_t rows{0};
    std::uint64_t input_bytes{0};
    std::uint64_t state_bytes{0};
    std::uint64_t output_bytes{0};
    std::uint64_t transfer_nanoseconds{0};
    std::uint64_t queue_nanoseconds{0};
    std::uint64_t compute_nanoseconds{0};
    std::uint64_t commit_nanoseconds{0};
    std::uint64_t elapsed_nanoseconds{0};
    std::uint64_t device_memory_bytes{0};
    std::uint64_t retries{0};
    std::uint64_t fallbacks{0};
};

class EngineCostObserver {
public:
    virtual ~EngineCostObserver() = default;
    virtual void observe(const EngineCostObservation& observation) noexcept = 0;
};

struct EngineCheckpoint {
    std::uint64_t state_generation{0};
    std::uint64_t topology_generation{0};
    std::vector<std::byte> payload;
};

class StatefulEngineBackend {
public:
    virtual ~StatefulEngineBackend() = default;

    [[nodiscard]] virtual StatefulEngineCapabilities engine_capabilities()
        const noexcept = 0;
    [[nodiscard]] virtual bindnordic::BackendCapabilities backend_capabilities()
        const noexcept = 0;
    virtual void initialise(
        const ProgramContract& contract,
        const ExecutionResources& resources) = 0;
    [[nodiscard]] virtual EngineAdvance advance(
        const TensorMap& state,
        const TensorMap& inputs,
        const EngineStep& step) = 0;
    [[nodiscard]] virtual std::vector<std::byte> checkpoint(
        const TensorMap& state) const;
    [[nodiscard]] virtual TensorMap restore(std::span<const std::byte> payload);
    virtual void close() noexcept = 0;
};

class TransactionalEngine final {
public:
    TransactionalEngine(
        ProgramContract contract,
        ExecutionResources resources,
        StatefulEngineBackend& backend,
        EngineCostObserver* observer = nullptr);
    ~TransactionalEngine();

    TransactionalEngine(const TransactionalEngine&) = delete;
    TransactionalEngine& operator=(const TransactionalEngine&) = delete;

    void initialise(TensorMap state, std::uint64_t topology_generation);
    [[nodiscard]] TensorMap advance(
        const TensorMap& inputs,
        const EngineStep& step);
    [[nodiscard]] EngineCheckpoint checkpoint() const;
    void restore(const EngineCheckpoint& checkpoint);
    void close() noexcept;

    [[nodiscard]] const TensorMap& state() const noexcept;
    [[nodiscard]] std::uint64_t state_generation() const noexcept;
    [[nodiscard]] std::uint64_t topology_generation() const noexcept;
    [[nodiscard]] bool active() const noexcept;

private:
    static void validate_tensors(const TensorMap& tensors, const char* kind);

    ProgramContract contract_;
    ExecutionResources resources_;
    StatefulEngineBackend& backend_;
    StatefulEngineCapabilities engine_capabilities_;
    bindnordic::BackendCapabilities backend_capabilities_;
    EngineCostObserver* observer_;
    TensorMap state_;
    std::uint64_t state_generation_{0};
    std::uint64_t topology_generation_{0};
    bool active_{false};
};

}  // namespace foamnordic::inference
