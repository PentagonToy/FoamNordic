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
#include <string>
#include <unordered_map>
#include <vector>

#include "foamnordic/backend/adapter/plan.hpp"
#include "foamnordic/backend/adapter/sequence.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace foamnordic::adapter {

struct ExchangeContract {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    void validate() const;
};

using InputFieldMap =
    std::unordered_map<std::string, fjord::TensorView>;
using OutputFieldMap =
    std::unordered_map<std::string, fjord::MutableTensorView>;

class AtomicFieldExchange {
public:
    AtomicFieldExchange(fjord::Harbor& harbor, ExchangeContract contract);

    void execute(
        std::uint64_t exchange_index,
        double physical_time,
        const MutableFieldMap& fields);
    void execute(
        std::uint64_t exchange_index,
        double physical_time,
        const InputFieldMap& inputs,
        const OutputFieldMap& outputs);

private:
    fjord::Harbor& harbor_;
    ExchangeContract contract_;
    std::uint64_t previous_exchange_{0};
    bool has_previous_exchange_{false};
};

class BlockingClosureExchange {
public:
    BlockingClosureExchange(fjord::Harbor& harbor, ExchangeContract contract);

    [[nodiscard]] std::uint64_t execute(
        std::uint64_t time_index,
        double physical_time,
        const MutableFieldMap& fields);
    [[nodiscard]] std::uint64_t execute(
        std::uint64_t time_index,
        double physical_time,
        const InputFieldMap& inputs,
        const OutputFieldMap& outputs);

private:
    AtomicFieldExchange exchange_;
    ExchangeSequence sequence_{ExchangeCadence::every_call};
};

}  // namespace foamnordic::adapter
