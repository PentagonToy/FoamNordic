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

#include "closureHook.H"

#include "fieldBridge.H"
#include "operations/frame.H"

#include <chrono>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace Foam::foamNordic {

std::vector<ClosureInput> ClosureHook::readInputs(const dictionary& dict) {
    const auto names = dict.get<wordList>("inputs");
    const auto expressions = dict.getOrDefault<List<string>>(
        "inputExpressions", List<string>());
    const auto patches = dict.getOrDefault<wordList>(
        "inputPatches", wordList());
    if (!expressions.empty() && expressions.size() != names.size()) {
        throw std::invalid_argument(
            "FoamNordic inputExpressions must have the same length as inputs.");
    }
    if (!patches.empty() && patches.size() != names.size()) {
        throw std::invalid_argument(
            "FoamNordic inputPatches must have the same length as inputs.");
    }

    std::vector<ClosureInput> result;
    result.reserve(names.size());
    std::unordered_set<std::string> unique;
    forAll(names, index) {
        const std::string key(names[index].c_str());
        const std::string expression(
            expressions.empty() ? names[index].c_str()
                                : expressions[index].c_str());
        if (key.empty() || expression.empty() || !unique.insert(key).second) {
            throw std::invalid_argument(
                "FoamNordic closure input keys and expressions must be "
                "non-empty, with unique keys.");
        }
        const word patch = patches.empty() || patches[index] == "none"
            ? word()
            : patches[index];
        if (!patch.empty()
            && expression.find_first_of("(),") != std::string::npos) {
            throw std::invalid_argument(
                "FoamNordic patch inputs must name a stored field, not a "
                "derived expression.");
        }
        result.push_back({key, expression, patch});
    }
    if (result.empty()) {
        throw std::invalid_argument(
            "FoamNordic closure requires at least one input expression.");
    }
    return result;
}

std::vector<ClosureOutput> ClosureHook::readOutputs(const dictionary& dict) {
    const auto fields = dict.get<wordList>("outputs");
    const auto keys = dict.getOrDefault<wordList>("outputKeys", wordList());
    const auto patches = dict.getOrDefault<wordList>(
        "outputPatches", wordList());
    if (fields.empty()) {
        throw std::invalid_argument(
            "FoamNordic closure requires at least one output field.");
    }
    if (!keys.empty() && keys.size() != fields.size()) {
        throw std::invalid_argument(
            "FoamNordic outputKeys must have the same length as outputs.");
    }
    if (!patches.empty() && patches.size() != fields.size()) {
        throw std::invalid_argument(
            "FoamNordic outputPatches must have the same length as outputs.");
    }
    std::vector<ClosureOutput> result;
    result.reserve(fields.size());
    std::unordered_set<std::string> unique;
    forAll(fields, index) {
        const std::string key(
            keys.empty() ? fields[index].c_str() : keys[index].c_str());
        if (key.empty() || !unique.insert(key).second) {
            throw std::invalid_argument(
                "FoamNordic closure output keys must be non-empty and unique.");
        }
        const word patch = patches.empty() || patches[index] == "none"
            ? word()
            : patches[index];
        result.push_back({key, fields[index], patch});
    }
    return result;
}

foamnordic::adapter::ExchangeContract ClosureHook::contract(
    const std::vector<ClosureInput>& inputs,
    const std::vector<ClosureOutput>& outputs) {
    foamnordic::adapter::ExchangeContract result;
    result.inputs.reserve(inputs.size());
    result.outputs.reserve(outputs.size());
    for (const auto& input : inputs) {
        result.inputs.push_back(input.key);
    }
    for (const auto& output : outputs) {
        result.outputs.push_back(output.key);
    }
    result.validate();
    return result;
}

ClosureHook::ClosureHook(const dictionary& dict)
    : inputs_(readInputs(dict)),
      outputs_(readOutputs(dict)),
      session_(dict, contract(inputs_, outputs_)),
      observation_(FieldProgramObservation::create(dict)) {}

ClosureHook::~ClosureHook() noexcept {
    try {
        shutdown();
    } catch (...) {
        // Destructors must not turn solver teardown into a failure path.
    }
}

std::uint64_t ClosureHook::invoke(const fvMesh& mesh, const Time& time) {
    return invoke(mesh, time, [](operations::OperationFrame&) {});
}

std::uint64_t ClosureHook::invoke(
    const fvMesh& mesh,
    const Time& time,
    const InputBinder& bindInputs) {
    operations::OperationFrame frame(mesh);
    if (bindInputs) {
        bindInputs(frame);
    }
    auto invocation = session_.begin(time);
    for (const auto& input : inputs_) {
        if (input.patch.empty()) {
            frame.provide(invocation, input.key, input.expression);
        } else {
            auto view = inputPatchView(
                mesh,
                word(input.expression),
                input.patch,
                0,
                0.0);
            view.name = input.key;
            invocation.provide(std::move(view));
        }
    }
    for (const auto& output : outputs_) {
        auto view = output.patch.empty()
            ? outputFieldView(mesh, output.field, 0, 0.0)
            : outputPatchView(mesh, output.field, output.patch, 0, 0.0);
        view.name = output.key;
        invocation.receive(std::move(view));
    }

    const auto closureStarted = std::chrono::steady_clock::now();
    const auto exchangeIndex = invocation.commit();
    const auto closureWait = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - closureStarted).count();
    for (const auto& output : outputs_) {
        if (output.patch.empty()) {
            correctFieldBoundary(mesh, output.field);
        }
    }
    if (observation_) {
        observation_->publish(mesh, time, exchangeIndex, closureWait);
    }
    return exchangeIndex;
}

void ClosureHook::shutdown() {
    if (observation_) {
        observation_->shutdown();
        observation_.reset();
    }
    session_.shutdown();
}

}  // namespace Foam::foamNordic
