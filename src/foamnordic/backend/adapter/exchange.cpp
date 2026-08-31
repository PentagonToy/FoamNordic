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

#include "foamnordic/backend/adapter/exchange.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace foamnordic::adapter {
namespace {

void validate_names(const std::vector<std::string>& names, const char* direction) {
    std::unordered_set<std::string> unique;
    for (const auto& name : names) {
        if (name.empty() || !unique.insert(name).second) {
            throw std::invalid_argument(
                std::string("FoamNordic ") + direction
                + " field names must be non-empty and unique.");
        }
    }
}

fjord::MutableTensorView require_field(
    const MutableFieldMap& fields,
    const std::string& name,
    std::uint64_t exchange_index,
    double physical_time) {
    const auto found = fields.find(name);
    if (found == fields.end()) {
        throw std::invalid_argument("FoamNordic exchange is missing field: " + name);
    }
    auto field = found->second;
    field.validate();
    if (field.time_index != exchange_index
        || std::abs(field.physical_time - physical_time) > 1.0e-12) {
        throw std::invalid_argument("FoamNordic exchange field metadata is out of sync: " + name);
    }
    return field;
}

}  // namespace

void ExchangeContract::validate() const {
    if (inputs.empty() || outputs.empty()) {
        throw std::invalid_argument(
            "A closure exchange requires at least one input and one output field.");
    }
    validate_names(inputs, "input");
    validate_names(outputs, "output");
}

AtomicFieldExchange::AtomicFieldExchange(
    fjord::Harbor& harbor,
    ExchangeContract contract)
    : harbor_(harbor), contract_(std::move(contract)) {
    contract_.validate();
}

BlockingClosureExchange::BlockingClosureExchange(
    fjord::Harbor& harbor,
    ExchangeContract contract)
    : exchange_(harbor, std::move(contract)) {}

std::uint64_t BlockingClosureExchange::execute(
    std::uint64_t time_index,
    double physical_time,
    const MutableFieldMap& fields) {
    const auto exchange_index = sequence_.next(time_index);
    if (!exchange_index) {
        throw std::logic_error(
            "Per-call closure sequencing did not issue an exchange index.");
    }
    auto rebound = fields;
    for (auto& [name, field] : rebound) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
    }
    exchange_.execute(*exchange_index, physical_time, rebound);
    return *exchange_index;
}

void AtomicFieldExchange::execute(
    std::uint64_t exchange_index,
    double physical_time,
    const MutableFieldMap& fields) {
    InputFieldMap inputs;
    OutputFieldMap outputs;
    for (const auto& name : contract_.inputs) {
        inputs.emplace(
            name,
            require_field(fields, name, exchange_index, physical_time)
                .read_only());
    }
    for (const auto& name : contract_.outputs) {
        outputs.emplace(
            name,
            require_field(fields, name, exchange_index, physical_time));
    }
    execute(exchange_index, physical_time, inputs, outputs);
}

void AtomicFieldExchange::execute(
    std::uint64_t exchange_index,
    double physical_time,
    const InputFieldMap& input_fields,
    const OutputFieldMap& output_fields) {
    if (!std::isfinite(physical_time)
        || (has_previous_exchange_ && exchange_index <= previous_exchange_)) {
        throw std::invalid_argument(
            "FoamNordic exchange identity must be finite and monotonically increasing.");
    }

    std::vector<fjord::TensorView> inputs;
    inputs.reserve(contract_.inputs.size());
    for (const auto& name : contract_.inputs) {
        const auto found = input_fields.find(name);
        if (found == input_fields.end()) {
            throw std::invalid_argument(
                "FoamNordic exchange is missing input field: " + name);
        }
        auto input = found->second;
        input.validate();
        if (input.time_index != exchange_index
            || input.solver_time_index
                   != input_fields.begin()->second.solver_time_index
            || std::abs(input.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument(
                "FoamNordic input field metadata is out of sync: " + name);
        }
        inputs.push_back(std::move(input));
    }
    harbor_.publish(exchange_index, inputs);

    std::unordered_map<std::string, fjord::Tensor> prepared;
    while (true) {
        auto message = harbor_.receive_message();
        if (message.kind == fjord::RuneKind::error) {
            throw std::runtime_error(
                "The native closure worker rejected the active exchange.");
        }
        if (message.kind == fjord::RuneKind::tensor) {
            if (!message.tensor || message.tensor->time_index != exchange_index
                || message.tensor->solver_time_index
                       != input_fields.begin()->second.solver_time_index
                || std::abs(
                       message.tensor->physical_time - physical_time)
                       > 1.0e-12
                || std::find(
                       contract_.outputs.begin(),
                       contract_.outputs.end(),
                       message.tensor->name)
                       == contract_.outputs.end()
                || prepared.contains(message.tensor->name)) {
                throw std::runtime_error(
                    "FoamNordic received an unexpected or duplicate prepared output.");
            }
            prepared.emplace(message.tensor->name, std::move(*message.tensor));
            continue;
        }
        if (message.kind != fjord::RuneKind::complete
            || message.exchange_index != exchange_index
            || message.tensor_count != prepared.size()
            || prepared.size() != contract_.outputs.size()) {
            throw std::runtime_error(
                "FoamNordic output batch was not atomically committed in full.");
        }
        break;
    }

    for (const auto& name : contract_.outputs) {
        const auto found = output_fields.find(name);
        if (found == output_fields.end()) {
            throw std::invalid_argument(
                "FoamNordic exchange is missing output field: " + name);
        }
        auto destination = found->second;
        destination.validate();
        if (destination.time_index != exchange_index
            || destination.solver_time_index
                   != input_fields.begin()->second.solver_time_index
            || std::abs(destination.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument(
                "FoamNordic output field metadata is out of sync: " + name);
        }
        const auto& source = prepared.at(name);
        if (source.element != destination.element
            || source.shape != destination.shape
            || source.bytes.size() != destination.bytes.size()) {
            throw std::runtime_error(
                "FoamNordic committed output does not match its field: " + name);
        }
    }
    for (const auto& name : contract_.outputs) {
        const auto destination = output_fields.at(name);
        const auto& source = prepared.at(name);
        std::memcpy(destination.bytes.data(), source.bytes.data(), source.bytes.size());
    }

    previous_exchange_ = exchange_index;
    has_previous_exchange_ = true;
}

std::uint64_t BlockingClosureExchange::execute(
    std::uint64_t time_index,
    double physical_time,
    const InputFieldMap& inputs,
    const OutputFieldMap& outputs) {
    const auto exchange_index = sequence_.next(time_index);
    if (!exchange_index) {
        throw std::logic_error(
            "Per-call closure sequencing did not issue an exchange index.");
    }
    auto rebound_inputs = inputs;
    for (auto& [name, field] : rebound_inputs) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
    }
    auto rebound_outputs = outputs;
    for (auto& [name, field] : rebound_outputs) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
    }
    exchange_.execute(
        *exchange_index,
        physical_time,
        rebound_inputs,
        rebound_outputs);
    return *exchange_index;
}

}  // namespace foamnordic::adapter
