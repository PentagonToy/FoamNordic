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

#include "foamnordic/backend/adapter/plan.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace foamnordic::adapter {

ExecutionPlan& ExecutionPlan::modify(std::string field, FieldTransform transform) {
    if (field.empty()) {
        throw std::invalid_argument("FoamNordic execution-plan field must not be empty.");
    }
    for (const auto& rule : rules_) {
        if (rule.field == field) {
            throw std::invalid_argument(
                "FoamNordic execution plan has multiple writers for field: " + field);
        }
    }
    rules_.push_back({std::move(field), std::move(transform)});
    return *this;
}

CompiledExecutionPlan ExecutionPlan::compile() const {
    if (rules_.empty()) {
        throw std::logic_error("Cannot compile an empty FoamNordic execution plan.");
    }
    return CompiledExecutionPlan(rules_);
}

CompiledExecutionPlan::CompiledExecutionPlan(std::vector<ExecutionPlan::Rule> rules)
    : rules_(std::move(rules)) {}

void CompiledExecutionPlan::execute(
    std::uint64_t exchange_index,
    double physical_time,
    const MutableFieldMap& fields) const {
    if (!std::isfinite(physical_time)) {
        throw std::invalid_argument("FoamNordic execution-plan physical time is invalid.");
    }
    for (const auto& rule : rules_) {
        const auto found = fields.find(rule.field);
        if (found == fields.end()) {
            throw std::invalid_argument(
                "FoamNordic execution plan is missing field: " + rule.field);
        }
        auto field = found->second;
        if (field.time_index != exchange_index
            || std::abs(field.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument("FoamNordic execution-plan metadata is out of sync.");
        }
        rule.transform.apply_in_place(field);
    }
}

std::size_t CompiledExecutionPlan::size() const noexcept { return rules_.size(); }

}  // namespace foamnordic::adapter
