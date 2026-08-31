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

#include "foamnordic/backend/adapter/exchange.hpp"

namespace foamnordic::adapter {

class ClosureInvocation {
public:
    ClosureInvocation(
        BlockingClosureExchange& exchange,
        std::uint64_t time_index,
        double physical_time);

    ClosureInvocation(const ClosureInvocation&) = delete;
    ClosureInvocation& operator=(const ClosureInvocation&) = delete;
    ClosureInvocation(ClosureInvocation&&) = delete;
    ClosureInvocation& operator=(ClosureInvocation&&) = delete;

    ClosureInvocation& provide(fjord::TensorView field);
    ClosureInvocation& receive(fjord::MutableTensorView field);
    [[nodiscard]] std::uint64_t commit();

private:
    BlockingClosureExchange& exchange_;
    std::uint64_t time_index_;
    double physical_time_;
    InputFieldMap inputs_;
    OutputFieldMap outputs_;
    bool committed_{false};
};

class ClosurePort {
public:
    ClosurePort(fjord::Harbor& harbor, ExchangeContract contract);

    [[nodiscard]] ClosureInvocation begin(
        std::uint64_t time_index,
        double physical_time);

private:
    BlockingClosureExchange exchange_;
};

}  // namespace foamnordic::adapter
