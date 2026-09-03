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

class FieldInvocation {
public:
    FieldInvocation(
        BlockingFieldExchange& exchange,
        std::uint64_t time_index,
        double physical_time);

    FieldInvocation(const FieldInvocation&) = delete;
    FieldInvocation& operator=(const FieldInvocation&) = delete;
    FieldInvocation(FieldInvocation&&) = delete;
    FieldInvocation& operator=(FieldInvocation&&) = delete;

    FieldInvocation& provide(fjord::TensorView field);
    FieldInvocation& receive(fjord::MutableTensorView field);
    [[nodiscard]] std::uint64_t commit();

private:
    BlockingFieldExchange& exchange_;
    std::uint64_t time_index_;
    double physical_time_;
    InputFieldMap inputs_;
    OutputFieldMap outputs_;
    bool committed_{false};
};

class FieldProgramPort {
public:
    FieldProgramPort(fjord::Harbor& harbor, ExchangeContract contract);

    [[nodiscard]] FieldInvocation begin(
        std::uint64_t time_index,
        double physical_time);

private:
    BlockingFieldExchange exchange_;
};

}  // namespace foamnordic::adapter
