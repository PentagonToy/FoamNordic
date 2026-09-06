#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "foamnordic/backend/inference/exchange.hpp"

namespace {

using foamnordic::inference::TensorMap;
using foamnordic::fjord::Element;
using foamnordic::fjord::Tensor;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Tensor scalar_field(
    std::string name,
    const std::vector<double>& values,
    std::uint64_t exchange_index = 1,
    double physical_time = 0.1) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        exchange_index,
        physical_time,
    };
}

Tensor float_scalar_field(
    std::string name,
    const std::vector<float>& values,
    std::uint64_t exchange_index = 1,
    double physical_time = 0.1) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        Element::float32,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        exchange_index,
        physical_time,
    };
}

std::vector<double> values_of(const Tensor& tensor) {
    std::vector<double> values(tensor.bytes.size() / sizeof(double));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

class CombustionBypass final : public foamnordic::inference::CellEvaluationPolicy {
public:
    std::vector<std::uint64_t> prepare(
        const TensorMap& inputs,
        TensorMap& outputs,
        std::uint64_t cell_count) const override {
        const auto variance = values_of(inputs.at("c_var"));
        std::vector<std::uint64_t> active;
        for (std::uint64_t cell = 0; cell < cell_count; ++cell) {
            if (variance[cell] > 0.05) {
                active.push_back(cell);
            }
        }
        outputs.emplace("omega_c", scalar_field("omega_c", std::vector<double>(cell_count, 0.0)));
        return active;
    }

    void merge(
        TensorMap& predictions,
        TensorMap& outputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t) const override {
        const auto predicted = values_of(predictions.at("omega_c"));
        auto merged = values_of(outputs.at("omega_c"));
        require(predicted.size() == active_cells.size(), "Active prediction size is incorrect.");
        for (std::size_t index = 0; index < active_cells.size(); ++index) {
            merged[active_cells[index]] = predicted[index];
        }
        outputs.insert_or_assign("omega_c", scalar_field("omega_c", merged));
    }
};

void test_combustion_bypass() {
    foamnordic::inference::CellInferenceExchange exchange({
        "progress-variable-combustion",
        {
            {"c_tilde", Element::float64, 1},
            {"c_var", Element::float64, 1},
            {"T_tilde", Element::float64, 1},
        },
        {{"omega_c", Element::float64, 1}},
    });
    exchange.begin(1, 0.1, 4);
    exchange.add_input(scalar_field("c_tilde", {0.1, 0.4, 0.7, 0.9}));
    exchange.add_input(scalar_field("c_var", {0.0, 0.1, 0.01, 0.2}));
    exchange.add_input(scalar_field("T_tilde", {300.0, 900.0, 1200.0, 1600.0}));
    exchange.seal_inputs();

    CombustionBypass bypass;
    const auto& active = exchange.prepare(bypass);
    require(active == std::vector<std::uint64_t>{1, 3}, "Combustion bypass mask is incorrect.");
    exchange.add_output(scalar_field("omega_c", {10.0, 20.0}));
    exchange.seal_outputs(bypass);
    const auto outputs = exchange.finish();
    require(exchange.inputs().empty(), "Finished exchange retained input payloads.");
    require(
        values_of(outputs.at("omega_c")) == std::vector<double>{0.0, 10.0, 0.0, 20.0},
        "Combustion bypass merge is incorrect.");
}

void test_incomplete_exchange_is_rejected() {
    foamnordic::inference::CellInferenceExchange exchange({
        "smagorinsky",
        {{"grad_U", Element::float64, 9}, {"delta", Element::float64, 1}},
        {{"nut", Element::float64, 1}},
    });
    exchange.begin(3, 0.2, 2);
    exchange.add_input(scalar_field("delta", {0.01, 0.02}, 3, 0.2));
    bool rejected = false;
    try {
        exchange.seal_inputs();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Incomplete LES closure input was accepted.");
    exchange.fail("Test failure cleanup");
    require(exchange.inputs().empty(), "Failed exchange retained input payloads.");
}

void test_in_place_field_contract() {
    foamnordic::inference::ProgramContract in_place{
        "velocity-correction",
        {{"U", Element::float64, 3}},
        {{"U", Element::float64, 3}},
    };
    in_place.validate();

    foamnordic::inference::ProgramContract duplicate_input{
        "invalid-velocity-correction",
        {{"U", Element::float64, 3}, {"U", Element::float64, 3}},
        {{"U", Element::float64, 3}},
    };
    bool rejected = false;
    try {
        duplicate_input.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Duplicate fields within one direction were accepted.");
}

void test_empty_rank_local_field_is_a_valid_exchange() {
    foamnordic::inference::CellInferenceExchange exchange({
        "empty-patch",
        {{"patch", Element::float64, 1}},
        {{"result", Element::float64, 1}},
    });
    exchange.begin(7, 0.5, 0);
    exchange.add_input(scalar_field("patch", {}, 7, 0.5));
    exchange.seal_inputs();
    foamnordic::inference::EvaluateAllCells policy;
    require(
        exchange.prepare(policy).empty(),
        "An empty rank-local patch produced active cells.");
    exchange.add_output(scalar_field("result", {}, 7, 0.5));
    exchange.seal_outputs(policy);
    require(
        exchange.finish().at("result").bytes.empty(),
        "An empty rank-local output gained payload bytes.");
}

void test_double_solver_input_is_accepted_by_float_model_contract() {
    foamnordic::inference::CellInferenceExchange exchange({
        "float-model",
        {{"delta", Element::float32, 1}},
        {{"nut", Element::float32, 1}},
    });
    exchange.begin(11, 0.75, 2);
    exchange.add_input(scalar_field("delta", {0.1, 0.2}, 11, 0.75));
    exchange.seal_inputs();
    foamnordic::inference::EvaluateAllCells policy;
    static_cast<void>(exchange.prepare(policy));
    exchange.add_output(float_scalar_field("nut", {0.01F, 0.02F}, 11, 0.75));
    exchange.seal_outputs(policy);
    require(
        exchange.finish().at("nut").element == Element::float32,
        "Float model output element was changed inside model inference.");
}

}  // namespace

int main() {
    test_combustion_bypass();
    test_incomplete_exchange_is_rejected();
    test_in_place_field_contract();
    test_empty_rank_local_field_is_a_valid_exchange();
    test_double_solver_input_is_accepted_by_float_model_contract();
}
