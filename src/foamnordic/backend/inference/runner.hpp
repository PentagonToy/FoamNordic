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

#include "foamnordic/backend/inference/closure.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace foamnordic::closure {

class ModelKernel {
public:
    virtual ~ModelKernel() = default;

    [[nodiscard]] virtual TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time) = 0;
};

class NativeClosureRunner {
public:
    NativeClosureRunner(
        fjord::Harbor& harbor,
        ClosureContract contract,
        const BypassPolicy& bypass,
        ModelKernel& kernel);

    [[nodiscard]] bool run_one();
    void run();

private:
    fjord::Harbor& harbor_;
    ExchangeMachine exchange_;
    const BypassPolicy& bypass_;
    ModelKernel& kernel_;
};

}  // namespace foamnordic::closure
