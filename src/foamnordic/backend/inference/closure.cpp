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

#include "foamnordic/backend/inference/closure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "foamnordic/runtime/log.hpp"

namespace foamnordic::closure {

void FieldContract::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("Closure field name must not be empty.");
    }
    if (fjord::element_size(element) == 0) {
        throw std::invalid_argument("Closure field element type is unsupported.");
    }
    if (components == 0) {
        throw std::invalid_argument("Closure field components must be positive.");
    }
}

void ClosureContract::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("Closure contract name must not be empty.");
    }
    if (inputs.empty() || outputs.empty()) {
        throw std::invalid_argument("Closure contract requires inputs and outputs.");
    }
    for (const auto* fields : {&inputs, &outputs}) {
        std::unordered_set<std::string> names;
        for (const auto& field : *fields) {
            field.validate();
            if (!names.insert(field.name).second) {
                throw std::invalid_argument(
                    "Closure field names must be unique within each direction.");
            }
        }
    }
}

std::vector<std::uint64_t> EvaluateEveryCell::prepare(
    const TensorMap&,
    TensorMap&,
    std::uint64_t cell_count) const {
    std::vector<std::uint64_t> cells(cell_count);
    std::iota(cells.begin(), cells.end(), 0);
    return cells;
}

void EvaluateEveryCell::merge(
    TensorMap& predictions,
    TensorMap& outputs,
    const std::vector<std::uint64_t>& active_cells,
    std::uint64_t cell_count) const {
    if (active_cells.size() != cell_count) {
        throw std::logic_error("EvaluateEveryCell requires every cell to be active.");
    }
    outputs = std::move(predictions);
}

ExchangeMachine::ExchangeMachine(ClosureContract contract) : contract_(std::move(contract)) {
    contract_.validate();
}

void ExchangeMachine::begin(
    std::uint64_t exchange_index,
    double physical_time,
    std::uint64_t cell_count,
    std::uint64_t solver_time_index) {
    if (state_ != ExchangeState::waiting && state_ != ExchangeState::completed) {
        throw std::logic_error("Closure exchange cannot begin from its current state.");
    }
    if (has_previous_exchange_ && exchange_index <= previous_exchange_index_) {
        throw std::invalid_argument("Closure exchange indices must increase monotonically.");
    }
    if (!std::isfinite(physical_time) || cell_count == 0) {
        throw std::invalid_argument("Closure exchange time and cell count are invalid.");
    }
    exchange_index_ = exchange_index;
    physical_time_ = physical_time;
    solver_time_index_ = solver_time_index;
    cell_count_ = cell_count;
    inputs_.clear();
    predictions_.clear();
    outputs_.clear();
    active_cells_.clear();
    failure_reason_.clear();
    state_ = ExchangeState::collecting_inputs;
}

const FieldContract& ExchangeMachine::require_field(
    const std::vector<FieldContract>& fields,
    const std::string& name,
    const char* direction) const {
    const auto field = std::find_if(fields.begin(), fields.end(), [&](const auto& candidate) {
        return candidate.name == name;
    });
    if (field == fields.end()) {
        throw std::invalid_argument(
            std::string("Unexpected closure ") + direction + " field: " + name);
    }
    return *field;
}

void ExchangeMachine::validate_tensor(
    const fjord::Tensor& tensor,
    const FieldContract& field,
    std::uint64_t expected_cells) const {
    tensor.view().validate();
    if (tensor.element != field.element || tensor.time_index != exchange_index_
        || tensor.solver_time_index != solver_time_index_
        || tensor.shape.empty() || tensor.shape.front() != expected_cells
        || std::abs(tensor.physical_time - physical_time_) > 1.0e-12) {
        throw std::invalid_argument("Closure tensor metadata does not match its exchange.");
    }
    std::uint64_t components = 1;
    for (std::size_t index = 1; index < tensor.shape.size(); ++index) {
        if (tensor.shape[index] != 0
            && components > std::numeric_limits<std::uint64_t>::max() / tensor.shape[index]) {
            throw std::overflow_error("Closure tensor component count overflowed.");
        }
        components *= tensor.shape[index];
    }
    if (components != field.components) {
        throw std::invalid_argument("Closure tensor component count is incorrect.");
    }
}

void ExchangeMachine::require_state(ExchangeState expected, const char* operation) const {
    if (state_ != expected) {
        throw std::logic_error(std::string(operation) + " is invalid in the current closure state.");
    }
}

void ExchangeMachine::add_input(fjord::Tensor tensor) {
    require_state(ExchangeState::collecting_inputs, "add_input");
    const auto& field = require_field(contract_.inputs, tensor.name, "input");
    validate_tensor(tensor, field, cell_count_);
    if (!inputs_.emplace(tensor.name, std::move(tensor)).second) {
        throw std::invalid_argument("Closure input field was supplied more than once.");
    }
}

void ExchangeMachine::seal_inputs() {
    require_state(ExchangeState::collecting_inputs, "seal_inputs");
    if (inputs_.size() != contract_.inputs.size()) {
        throw std::logic_error("Closure inputs are incomplete.");
    }
    state_ = ExchangeState::inputs_ready;
}

const std::vector<std::uint64_t>& ExchangeMachine::prepare(const BypassPolicy& policy) {
    require_state(ExchangeState::inputs_ready, "prepare");
    active_cells_ = policy.prepare(inputs_, outputs_, cell_count_);
    if (!std::is_sorted(active_cells_.begin(), active_cells_.end())
        || std::adjacent_find(active_cells_.begin(), active_cells_.end()) != active_cells_.end()
        || (!active_cells_.empty() && active_cells_.back() >= cell_count_)) {
        throw std::runtime_error("Closure bypass policy returned invalid active cells.");
    }
    state_ = ExchangeState::evaluating;
    return active_cells_;
}

void ExchangeMachine::add_output(fjord::Tensor tensor) {
    require_state(ExchangeState::evaluating, "add_output");
    const auto& field = require_field(contract_.outputs, tensor.name, "output");
    validate_tensor(tensor, field, active_cells_.size());
    predictions_.insert_or_assign(tensor.name, std::move(tensor));
}

void ExchangeMachine::seal_outputs(const BypassPolicy& policy) {
    require_state(ExchangeState::evaluating, "seal_outputs");
    policy.merge(predictions_, outputs_, active_cells_, cell_count_);
    if (outputs_.size() != contract_.outputs.size()) {
        throw std::logic_error("Closure outputs are incomplete after inference and bypass.");
    }
    for (const auto& field : contract_.outputs) {
        const auto output = outputs_.find(field.name);
        if (output == outputs_.end()) {
            throw std::logic_error("Closure bypass did not produce every required output.");
        }
        validate_tensor(output->second, field, cell_count_);
    }
    state_ = ExchangeState::outputs_ready;
}

TensorMap ExchangeMachine::finish() {
    require_state(ExchangeState::outputs_ready, "finish");
    state_ = ExchangeState::completed;
    previous_exchange_index_ = exchange_index_;
    has_previous_exchange_ = true;
    return std::move(outputs_);
}

void ExchangeMachine::fail(std::string reason) {
    if (reason.empty()) {
        reason = "Unknown closure failure.";
    }
    failure_reason_ = std::move(reason);
    state_ = ExchangeState::failed;
    native::log(native::LogLevel::error, failure_reason_);
}

ExchangeState ExchangeMachine::state() const noexcept { return state_; }
std::uint64_t ExchangeMachine::exchange_index() const noexcept { return exchange_index_; }
double ExchangeMachine::physical_time() const noexcept { return physical_time_; }
std::uint64_t ExchangeMachine::solver_time_index() const noexcept {
    return solver_time_index_;
}
std::uint64_t ExchangeMachine::cell_count() const noexcept { return cell_count_; }
const TensorMap& ExchangeMachine::inputs() const noexcept { return inputs_; }
const std::string& ExchangeMachine::failure_reason() const noexcept { return failure_reason_; }

}  // namespace foamnordic::closure
