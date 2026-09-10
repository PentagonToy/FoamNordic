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
#ifndef Foam_foamNordic_progressRateFjord_H
#define Foam_foamNordic_progressRateFjord_H

#include "closureHook.H"
#include "volFields.H"

namespace Foam::foamNordic {

class progressRateFjord final {
    const volScalarField& progress_;
    const volScalarField& variance_;
    const volScalarField& temperature_;
    volScalarField& reactionRate_;
    volScalarField normalizedVariance_;
    ClosureHook closure_;

    [[nodiscard]] static const dictionary& validateContract(
        const dictionary& dictionary,
        const volScalarField& reactionRate);

public:
    progressRateFjord(
        const dictionary& dictionary,
        const volScalarField& progress,
        const volScalarField& variance,
        const volScalarField& temperature,
        volScalarField& reactionRate);

    progressRateFjord(const progressRateFjord&) = delete;
    progressRateFjord& operator=(const progressRateFjord&) = delete;

    void correct();
    void shutdown();
};

}  // namespace Foam::foamNordic

#endif
