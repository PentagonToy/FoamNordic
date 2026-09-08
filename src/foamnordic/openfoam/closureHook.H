/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider: ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-only
 */
#ifndef Foam_foamNordic_closureHook_H
#define Foam_foamNordic_closureHook_H

#include "fieldProgramSession.H"
#include "fieldProgramObservation.H"

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
    word patch;
};

struct ClosureOutput final {
    std::string key;
    word field;
    word patch;
};

class ClosureHook final {
    std::vector<ClosureInput> inputs_;
    std::vector<ClosureOutput> outputs_;
    FieldProgramSession session_;
    std::unique_ptr<FieldProgramObservation> observation_;
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
