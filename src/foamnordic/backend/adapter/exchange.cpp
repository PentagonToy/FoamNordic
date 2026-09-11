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
#include <chrono>
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

enum class CopyKind {
    direct,
    widen_float,
    narrow_float,
};

struct PreparedCopy {
    const fjord::Tensor* source;
    fjord::MutableTensorView destination;
    CopyKind kind;
};

[[nodiscard]] CopyKind copy_kind(
    const fjord::Tensor& source,
    const fjord::MutableTensorView& destination) {
    if (source.element == destination.element) {
        return CopyKind::direct;
    }
    if (source.element == fjord::Element::float32
        && destination.element == fjord::Element::float64) {
        return CopyKind::widen_float;
    }
    if (source.element == fjord::Element::float64
        && destination.element == fjord::Element::float32) {
        return CopyKind::narrow_float;
    }
    throw std::runtime_error(
        "FoamNordic cannot convert the committed model output element type.");
}

void copy_values(
    const fjord::Tensor& source,
    const fjord::MutableTensorView& destination,
    CopyKind kind) noexcept {
    const auto count = source.view().element_count();
    if (kind == CopyKind::direct) {
        std::memcpy(
            destination.bytes.data(), source.bytes.data(), source.bytes.size());
        return;
    }
    if (kind == CopyKind::widen_float) {
        const auto* input = reinterpret_cast<const float*>(source.bytes.data());
        auto* output = reinterpret_cast<double*>(destination.bytes.data());
        std::transform(input, input + count, output, [](float value) {
            return static_cast<double>(value);
        });
        return;
    }
    const auto* input = reinterpret_cast<const double*>(source.bytes.data());
    auto* output = reinterpret_cast<float*>(destination.bytes.data());
    std::transform(input, input + count, output, [](double value) {
        return static_cast<float>(value);
    });
}

}  // namespace

void ExchangeContract::validate() const {
    if (inputs.empty() || outputs.empty()) {
        throw std::invalid_argument(
            "A model exchange requires at least one input and one output field.");
    }
    validate_names(inputs, "input");
    validate_names(outputs, "output");
}

AtomicFieldExchange::AtomicFieldExchange(
    fjord::Harbor& harbor,
    ExchangeContract contract,
    std::uint64_t profile_every)
    : harbor_(harbor),
      contract_(std::move(contract)),
      profile_every_(profile_every) {
    contract_.validate();
}

BlockingFieldExchange::BlockingFieldExchange(
    fjord::Harbor& harbor,
    ExchangeContract contract,
    std::uint64_t profile_every)
    : exchange_(harbor, std::move(contract), profile_every) {}

std::uint64_t BlockingFieldExchange::execute(
    std::uint64_t time_index,
    double physical_time,
    const MutableFieldMap& fields,
    std::uint64_t topology_generation) {
    const auto exchange_index = sequence_.next(time_index);
    if (!exchange_index) {
        throw std::logic_error(
            "Per-call model sequencing did not issue an exchange index.");
    }
    auto rebound = fields;
    for (auto& [name, field] : rebound) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
        field.topology_generation = topology_generation;
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
            || input.topology_generation
                   != input_fields.begin()->second.topology_generation
            || std::abs(input.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument(
                "FoamNordic input field metadata is out of sync: " + name);
        }
        inputs.push_back(std::move(input));
    }
    const auto sampled = profile_every_ != 0
        && profile_sequence_++ % profile_every_ == 0;
    const auto round_trip_started = sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    harbor_.publish_request(exchange_index, inputs);
    const auto request_published = sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    std::unordered_map<std::string, fjord::Tensor> prepared;
    while (true) {
        auto message = harbor_.receive_message();
        if (message.kind == fjord::RuneKind::error) {
            throw std::runtime_error(
                "The native model worker rejected the active exchange.");
        }
        if (message.kind == fjord::RuneKind::tensor) {
            if (!message.tensor || message.tensor->time_index != exchange_index
                || message.tensor->solver_time_index
                       != input_fields.begin()->second.solver_time_index
                || message.tensor->topology_generation
                       != input_fields.begin()->second.topology_generation
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
            || message.topology_generation
                   != input_fields.begin()->second.topology_generation
            || message.tensor_count != prepared.size()
            || prepared.size() != contract_.outputs.size()) {
            throw std::runtime_error(
                "FoamNordic output batch was not atomically committed in full.");
        }
        break;
    }
    const auto response_received = sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    std::vector<PreparedCopy> copies;
    copies.reserve(contract_.outputs.size());
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
            || destination.topology_generation
                   != input_fields.begin()->second.topology_generation
            || std::abs(destination.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument(
                "FoamNordic output field metadata is out of sync: " + name);
        }
        const auto& source = prepared.at(name);
        const auto source_count = source.view().element_count();
        const auto destination_count = destination.read_only().element_count();
        if (source.shape != destination.shape
            || source_count != destination_count) {
            throw std::runtime_error(
                "FoamNordic committed output does not match its field: " + name);
        }
        const auto destination_begin = reinterpret_cast<std::uintptr_t>(
            destination.bytes.data());
        const auto destination_end = destination_begin + destination.bytes.size();
        for (const auto& copy : copies) {
            const auto other_begin = reinterpret_cast<std::uintptr_t>(
                copy.destination.bytes.data());
            const auto other_end = other_begin + copy.destination.bytes.size();
            if (destination_begin < other_end && other_begin < destination_end) {
                throw std::runtime_error(
                    "FoamNordic committed outputs alias solver-owned storage.");
            }
        }
        copies.push_back({&source, destination, copy_kind(source, destination)});
    }
    for (const auto& copy : copies) {
        copy_values(*copy.source, copy.destination, copy.kind);
    }
    if (sampled) {
        const auto field_committed = std::chrono::steady_clock::now();
        const auto nanoseconds = [](auto duration) {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                    .count());
        };
        ++profile_.samples;
        profile_.round_trip_nanoseconds +=
            nanoseconds(response_received - round_trip_started);
        profile_.request_publish_nanoseconds +=
            nanoseconds(request_published - round_trip_started);
        profile_.response_wait_nanoseconds +=
            nanoseconds(response_received - request_published);
        profile_.field_commit_nanoseconds +=
            nanoseconds(field_committed - response_received);
    }

    previous_exchange_ = exchange_index;
    has_previous_exchange_ = true;
}

SolverExchangeSnapshot AtomicFieldExchange::profile_snapshot() const noexcept {
    return profile_;
}

std::uint64_t BlockingFieldExchange::execute(
    std::uint64_t time_index,
    double physical_time,
    const InputFieldMap& inputs,
    const OutputFieldMap& outputs,
    std::uint64_t topology_generation) {
    const auto exchange_index = sequence_.next(time_index);
    if (!exchange_index) {
        throw std::logic_error(
            "Per-call model sequencing did not issue an exchange index.");
    }
    auto rebound_inputs = inputs;
    for (auto& [name, field] : rebound_inputs) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
        field.topology_generation = topology_generation;
    }
    auto rebound_outputs = outputs;
    for (auto& [name, field] : rebound_outputs) {
        static_cast<void>(name);
        field.time_index = *exchange_index;
        field.physical_time = physical_time;
        field.solver_time_index = time_index;
        field.topology_generation = topology_generation;
    }
    exchange_.execute(
        *exchange_index,
        physical_time,
        rebound_inputs,
        rebound_outputs);
    return *exchange_index;
}

SolverExchangeSnapshot BlockingFieldExchange::profile_snapshot() const noexcept {
    return exchange_.profile_snapshot();
}

}  // namespace foamnordic::adapter
