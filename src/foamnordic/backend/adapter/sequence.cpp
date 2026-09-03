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

#include "foamnordic/backend/adapter/sequence.hpp"

#include <stdexcept>

namespace foamnordic::adapter {

ExchangeSequence::ExchangeSequence(ExchangeCadence cadence)
    : cadence_(cadence) {}

void ExchangeSequence::configure(ExchangeCadence cadence) {
    if (started_ && cadence != cadence_) {
        throw std::logic_error(
            "FoamNordic exchange cadence cannot change after execution starts.");
    }
    cadence_ = cadence;
}

std::optional<std::uint64_t> ExchangeSequence::next(
    std::uint64_t time_index) {
    if (started_ && time_index < previous_time_index_) {
        throw std::invalid_argument(
            "OpenFOAM time index moved backwards during model exchange.");
    }
    if (cadence_ == ExchangeCadence::time_step) {
        if (started_ && time_index == previous_time_index_) {
            return std::nullopt;
        }
        previous_time_index_ = time_index;
        started_ = true;
        return time_index;
    }
    if (started_ && next_call_exchange_ == 0) {
        throw std::overflow_error(
            "FoamNordic per-call model exchange index overflowed.");
    }
    previous_time_index_ = time_index;
    started_ = true;
    return next_call_exchange_++;
}

ExchangeCadence ExchangeSequence::cadence() const noexcept {
    return cadence_;
}

}  // namespace foamnordic::adapter
