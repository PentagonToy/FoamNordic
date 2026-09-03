/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider:  ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef Foam_foamNordic_fieldProgramObservation_H
#define Foam_foamNordic_fieldProgramObservation_H

#include "Time.H"
#include "dictionary.H"
#include "fvMesh.H"
#include "wordList.H"

#include "foamnordic/backend/adapter/observation.hpp"
#include "foamnordic/backend/adapter/observation_stream.hpp"

#include <cstdint>
#include <memory>

namespace Foam::foamNordic {

class FieldProgramObservation final {
    wordList fields_;
    foamnordic::adapter::CompiledObservationPlan plan_;
    std::unique_ptr<foamnordic::adapter::ObservationPublisher> publisher_;
    std::unique_ptr<foamnordic::adapter::ObservationJsonlWriter> writer_;
    bool enabled_{true};

    explicit FieldProgramObservation(const dictionary& dict);

public:
    [[nodiscard]] static std::unique_ptr<FieldProgramObservation> create(
        const dictionary& closureDict);

    FieldProgramObservation(const FieldProgramObservation&) = delete;
    FieldProgramObservation& operator=(const FieldProgramObservation&) = delete;

    void publish(
        const fvMesh& mesh,
        const Time& time,
        std::uint64_t exchangeIndex,
        double closureWait) noexcept;
    void shutdown() noexcept;
};

}  // namespace Foam::foamNordic

#endif
