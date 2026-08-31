/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider: ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "progressVariableSource.H"

#include "fvMatrices.H"
#include "volFields.H"

#include <stdexcept>

namespace Foam::foamNordic::combustion {

tmp<fvScalarMatrix> explicitSource(
    volScalarField& progress,
    const volScalarField& reactionRate) {
    if (&progress.mesh() != &reactionRate.mesh()) {
        throw std::invalid_argument(
            "FoamNordic progress and reaction-rate fields must share a mesh.");
    }

    auto matrix = tmp<fvScalarMatrix>::New(
        progress,
        reactionRate.dimensions() * dimVolume);

    // fvMatrix::operator-= adds a field to the matrix source, which is the
    // positive right-hand-side convention used by OpenFOAM's operator==.
    matrix.ref() -= reactionRate;
    return matrix;
}

tmp<fvScalarMatrix> explicitSpecificSource(
    volScalarField& progress,
    const volScalarField& reactionRate,
    const volScalarField& density) {
    if (&progress.mesh() != &reactionRate.mesh()
        || &progress.mesh() != &density.mesh()) {
        throw std::invalid_argument(
            "FoamNordic progress, reaction-rate, and density fields must "
            "share a mesh.");
    }
    if (reactionRate.dimensions() != dimless / dimTime) {
        throw std::invalid_argument(
            "FoamNordic specific reaction rate must have dimensions 1/time.");
    }
    if (density.dimensions() != dimDensity) {
        throw std::invalid_argument(
            "FoamNordic density must have mass/volume dimensions.");
    }

    const tmp<volScalarField> volumetricRate = density * reactionRate;
    return explicitSource(progress, volumetricRate());
}

}  // namespace Foam::foamNordic::combustion
