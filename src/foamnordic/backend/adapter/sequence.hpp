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
#include <optional>

namespace foamnordic::adapter {

enum class ExchangeCadence { time_step, every_call };

class ExchangeSequence {
public:
    explicit ExchangeSequence(
        ExchangeCadence cadence = ExchangeCadence::time_step);

    void configure(ExchangeCadence cadence);
    [[nodiscard]] std::optional<std::uint64_t> next(
        std::uint64_t time_index);
    [[nodiscard]] ExchangeCadence cadence() const noexcept;

private:
    ExchangeCadence cadence_;
    std::uint64_t previous_time_index_{0};
    std::uint64_t next_call_exchange_{0};
    bool started_{false};
};

}  // namespace foamnordic::adapter
