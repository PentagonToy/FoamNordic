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

#ifndef Foam_foamNordic_closureObservation_H
#define Foam_foamNordic_closureObservation_H

#include "Time.H"
#include "dictionary.H"
#include "fvMesh.H"
#include "wordList.H"

#include "foamnordic/backend/adapter/observation.hpp"
#include "foamnordic/backend/adapter/observation_stream.hpp"

#include <cstdint>
#include <memory>

namespace Foam::foamNordic {

class ClosureObservation final {
    wordList fields_;
    foamnordic::adapter::CompiledObservationPlan plan_;
    std::unique_ptr<foamnordic::adapter::ObservationPublisher> publisher_;
    std::unique_ptr<foamnordic::adapter::ObservationJsonlWriter> writer_;
    bool enabled_{true};

    explicit ClosureObservation(const dictionary& dict);

public:
    [[nodiscard]] static std::unique_ptr<ClosureObservation> create(
        const dictionary& closureDict);

    ClosureObservation(const ClosureObservation&) = delete;
    ClosureObservation& operator=(const ClosureObservation&) = delete;

    void publish(
        const fvMesh& mesh,
        const Time& time,
        std::uint64_t exchangeIndex,
        double closureWait) noexcept;
    void shutdown() noexcept;
};

}  // namespace Foam::foamNordic

#endif
