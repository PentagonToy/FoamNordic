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
        ModelKernel& kernel);

    [[nodiscard]] bool run_one();
    void run();

private:
    fjord::Harbor& harbor_;
    CellInferenceExchange exchange_;
    const CellEvaluationPolicy& bypass_;
    ModelKernel& kernel_;
};

}  // namespace foamnordic::inference
