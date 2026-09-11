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
#include <vector>

#include "foamnordic/backend/inference/exchange.hpp"
#include "foamnordic/backend/inference/execution.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace foamnordic::inference {

class ModelKernel {
public:
    virtual ~ModelKernel() = default;

    // False keeps the conservative worker-wide serialization used for
    // arbitrary stateful kernels. A kernel returning true must synchronize
    // only its non-reentrant backend resources itself.
    [[nodiscard]] virtual bool owns_backend_synchronization() const noexcept {
        return false;
    }

    [[nodiscard]] virtual ExecutionSnapshot execution_snapshot() const {
        return {};
    }

    [[nodiscard]] virtual bool telemetry_enabled() const noexcept {
        return false;
    }

    virtual void observe_exchange(
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t) noexcept {}

    virtual void cancel_pending() noexcept {}

    [[nodiscard]] virtual TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) = 0;
};

class InferenceRunner {
public:
    InferenceRunner(
        fjord::Harbor& harbor,
        ProgramContract contract,
        const CellEvaluationPolicy& bypass,
        ModelKernel& kernel,
        std::uint64_t maximum_exchange_payload =
            16ULL * 1024ULL * 1024ULL * 1024ULL,
        ReceiveCreditPool* receive_credits = nullptr,
        ReceiveCreditPool* output_credits = nullptr);

    [[nodiscard]] bool run_one();
    void run();

private:
    fjord::Harbor& harbor_;
    CellInferenceExchange exchange_;
    const CellEvaluationPolicy& bypass_;
    ModelKernel& kernel_;
    std::uint64_t maximum_exchange_payload_;
    ReceiveCreditPool* receive_credits_;
    ReceiveCreditPool* output_credits_;
    std::uint64_t output_bytes_per_cell_{0};
    bool telemetry_enabled_{false};
};

}  // namespace foamnordic::inference
