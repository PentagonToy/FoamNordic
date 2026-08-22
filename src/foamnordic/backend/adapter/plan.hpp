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

#include <string>
#include <unordered_map>
#include <vector>

#include "foamnordic/backend/adapter/field.hpp"

namespace foamnordic::adapter {

using MutableFieldMap = std::unordered_map<std::string, fjord::MutableTensorView>;

class CompiledExecutionPlan;

class ExecutionPlan {
public:
    ExecutionPlan& modify(std::string field, FieldTransform transform);
    [[nodiscard]] CompiledExecutionPlan compile() const;

private:
    struct Rule {
        std::string field;
        FieldTransform transform;
    };

    std::vector<Rule> rules_;

    friend class CompiledExecutionPlan;
};

class CompiledExecutionPlan {
public:
    void execute(
        std::uint64_t exchange_index,
        double physical_time,
        const MutableFieldMap& fields) const;

    [[nodiscard]] std::size_t size() const noexcept;

private:
    explicit CompiledExecutionPlan(std::vector<ExecutionPlan::Rule> rules);
    std::vector<ExecutionPlan::Rule> rules_;

    friend class ExecutionPlan;
};

}  // namespace foamnordic::adapter
