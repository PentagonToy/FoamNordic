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

#include "foamnordic/backend/adapter/port.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace foamnordic::adapter {

FieldInvocation::FieldInvocation(
    BlockingFieldExchange& exchange,
    std::uint64_t time_index,
    double physical_time)
    : exchange_(exchange),
      time_index_(time_index),
      physical_time_(physical_time) {
    if (!std::isfinite(physical_time_)) {
        throw std::invalid_argument(
            "FoamNordic model invocation physical time must be finite.");
    }
}

FieldInvocation& FieldInvocation::provide(fjord::TensorView field) {
    field.validate();
    if (committed_ || !inputs_.emplace(field.name, std::move(field)).second) {
        throw std::logic_error(
            "FoamNordic model input was duplicated or provided after commit.");
    }
    return *this;
}

FieldInvocation& FieldInvocation::receive(
    fjord::MutableTensorView field) {
    field.validate();
    if (committed_ || !outputs_.emplace(field.name, std::move(field)).second) {
        throw std::logic_error(
            "FoamNordic model output was duplicated or received after commit.");
    }
    return *this;
}

std::uint64_t FieldInvocation::commit() {
    if (committed_) {
        throw std::logic_error(
            "FoamNordic model invocation was committed more than once.");
    }
    const auto exchange_index =
        exchange_.execute(time_index_, physical_time_, inputs_, outputs_);
    committed_ = true;
    return exchange_index;
}

FieldProgramPort::FieldProgramPort(
    fjord::Harbor& harbor,
    ExchangeContract contract)
    : exchange_(harbor, std::move(contract)) {}

FieldInvocation FieldProgramPort::begin(
    std::uint64_t time_index,
    double physical_time) {
    return FieldInvocation(exchange_, time_index, physical_time);
}

}  // namespace foamnordic::adapter
