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

#include "foamnordic/fjord/tensor.hpp"

namespace foamnordic::inference {

struct FieldContract {
    std::string name;
    fjord::Element element{fjord::Element::float64};
    std::uint64_t components{1};

    void validate() const;
};

struct ProgramContract {
    std::string name;
    std::vector<FieldContract> inputs;
    std::vector<FieldContract> outputs;

    void validate() const;
};

using TensorMap = std::unordered_map<std::string, fjord::Tensor>;

class CellEvaluationPolicy {
public:
    virtual ~CellEvaluationPolicy() = default;

    [[nodiscard]] virtual std::vector<std::uint64_t> prepare(
        const TensorMap& inputs,
        TensorMap& outputs,
        std::uint64_t cell_count) const = 0;

    virtual void merge(
        TensorMap& predictions,
        TensorMap& outputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t cell_count) const = 0;
};

class EvaluateAllCells final : public CellEvaluationPolicy {
public:
    [[nodiscard]] std::vector<std::uint64_t> prepare(
        const TensorMap& inputs,
        TensorMap& outputs,
        std::uint64_t cell_count) const override;

    void merge(
        TensorMap& predictions,
        TensorMap& outputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t cell_count) const override;
};

enum class ExchangeState {
    waiting,
    collecting_inputs,
    inputs_ready,
    evaluating,
    outputs_ready,
    completed,
    failed,
};

class CellInferenceExchange {
public:
    explicit CellInferenceExchange(ProgramContract contract);

    void begin(
        std::uint64_t exchange_index,
        double physical_time,
        std::uint64_t cell_count,
        std::uint64_t solver_time_index = 0);
    void add_input(fjord::Tensor tensor);
    void seal_inputs();
    [[nodiscard]] const std::vector<std::uint64_t>& prepare(const CellEvaluationPolicy& policy);
    void add_output(fjord::Tensor tensor);
    void seal_outputs(const CellEvaluationPolicy& policy);
    [[nodiscard]] TensorMap finish();
    void fail(std::string reason);

    [[nodiscard]] ExchangeState state() const noexcept;
    [[nodiscard]] std::uint64_t exchange_index() const noexcept;
    [[nodiscard]] double physical_time() const noexcept;
    [[nodiscard]] std::uint64_t solver_time_index() const noexcept;
    [[nodiscard]] std::uint64_t cell_count() const noexcept;
    [[nodiscard]] const TensorMap& inputs() const noexcept;
    [[nodiscard]] const std::string& failure_reason() const noexcept;

private:
    void require_state(ExchangeState expected, const char* operation) const;
    void validate_tensor(
        const fjord::Tensor& tensor,
        const FieldContract& field,
        std::uint64_t expected_cells) const;
    [[nodiscard]] const FieldContract& require_field(
        const std::vector<FieldContract>& fields,
        const std::string& name,
        const char* direction) const;

    ProgramContract contract_;
    ExchangeState state_{ExchangeState::waiting};
    std::uint64_t exchange_index_{0};
    std::uint64_t previous_exchange_index_{0};
    bool has_previous_exchange_{false};
    double physical_time_{0.0};
    std::uint64_t solver_time_index_{0};
    std::uint64_t cell_count_{0};
    TensorMap inputs_;
    TensorMap predictions_;
    TensorMap outputs_;
    std::vector<std::uint64_t> active_cells_;
    std::string failure_reason_;
};

}  // namespace foamnordic::inference
