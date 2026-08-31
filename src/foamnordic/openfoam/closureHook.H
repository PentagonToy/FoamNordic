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

#ifndef Foam_foamNordic_closureHook_H
#define Foam_foamNordic_closureHook_H

#include "closureSession.H"
#include "closureObservation.H"

#include "fvMesh.H"
#include "wordList.H"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Foam::foamNordic {

namespace operations {
class OperationFrame;
}

struct ClosureInput final {
    std::string key;
    std::string expression;
};

struct ClosureOutput final {
    std::string key;
    word field;
};

class ClosureHook final {
    std::vector<ClosureInput> inputs_;
    std::vector<ClosureOutput> outputs_;
    ClosureSession session_;
    std::unique_ptr<ClosureObservation> observation_;
    bool profileEnabled_{false};
    bool profileReported_{false};
    std::uint64_t profileCalls_{0};
    double profileWait_{0.0};
    double profileMaximumWait_{0.0};

    [[nodiscard]] static std::vector<ClosureInput> readInputs(
        const dictionary& dict);
    [[nodiscard]] static std::vector<ClosureOutput> readOutputs(
        const dictionary& dict);
    [[nodiscard]] static foamnordic::adapter::ExchangeContract contract(
        const std::vector<ClosureInput>& inputs,
        const std::vector<ClosureOutput>& outputs);

public:
    using InputBinder = std::function<void(operations::OperationFrame&)>;

    explicit ClosureHook(const dictionary& dict);
    ~ClosureHook() noexcept;

    ClosureHook(const ClosureHook&) = delete;
    ClosureHook& operator=(const ClosureHook&) = delete;

    [[nodiscard]] std::uint64_t invoke(
        const fvMesh& mesh,
        const Time& time);
    [[nodiscard]] std::uint64_t invoke(
        const fvMesh& mesh,
        const Time& time,
        const InputBinder& bindInputs);

    void shutdown();
};

}  // namespace Foam::foamNordic

#endif
