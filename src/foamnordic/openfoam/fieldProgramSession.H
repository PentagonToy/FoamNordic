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
#ifndef Foam_foamNordic_fieldProgramSession_H
#define Foam_foamNordic_fieldProgramSession_H

#include "Time.H"
#include "dictionary.H"
#include "fvMesh.H"
#include "string.H"
#include "word.H"

#include "foamnordic/backend/adapter/port.hpp"

#include <cstdint>
#include <memory>

namespace foamnordic::fjord {
class Harbor;
}

namespace Foam::foamNordic {

[[nodiscard]] string resolveRankAddress(const string& address);

[[nodiscard]] std::unique_ptr<foamnordic::fjord::Harbor> connectSession(
    const string& address,
    bool sharedMemory,
    bool ucx,
    std::uint64_t sessionId);

void writeSolverTelemetry(
    const word& program,
    const foamnordic::adapter::SolverExchangeSnapshot& snapshot);

class MeshTopologySequence final {
    string facesInstance_;
    label cells_{-1};
    label faces_{-1};
    label points_{-1};
    std::uint64_t generation_{0};
    bool initialised_{false};
    bool wasChanging_{false};
    bool explicitlyChanged_{false};

public:
    [[nodiscard]] std::uint64_t observe(const fvMesh& mesh);
    void markChanged() noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
};

class FieldProgramSession final {
    std::unique_ptr<foamnordic::fjord::Harbor> harbor_;
    std::unique_ptr<foamnordic::adapter::FieldProgramPort> port_;
    MeshTopologySequence topology_;

public:
    FieldProgramSession(
        const dictionary& dict,
        foamnordic::adapter::ExchangeContract contract);
    ~FieldProgramSession();

    FieldProgramSession(const FieldProgramSession&) = delete;
    FieldProgramSession& operator=(const FieldProgramSession&) = delete;

    [[nodiscard]] foamnordic::adapter::FieldInvocation begin(
        const fvMesh& mesh,
        const Time& time);
    [[nodiscard]] std::uint64_t topologyGeneration() const noexcept;

    [[nodiscard]] foamnordic::adapter::SolverExchangeSnapshot
    profileSnapshot() const noexcept;

    void shutdown();
};

}  // namespace Foam::foamNordic

#endif
