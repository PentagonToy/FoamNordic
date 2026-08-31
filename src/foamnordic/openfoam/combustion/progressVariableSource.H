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

#ifndef Foam_foamNordic_progressVariableSource_H
#define Foam_foamNordic_progressVariableSource_H

#include "fvMatricesFwd.H"
#include "volFieldsFwd.H"

namespace Foam::foamNordic::combustion {

//- Return a matrix containing one explicit, positive, volumetric source.
//  The returned matrix is intended for the right-hand side of a solver-owned
//  progress-variable equation:
//
//      transport == combustion->R(progress)
//
//  OpenFOAM checks the equation dimensions when the matrices are combined.
tmp<fvScalarMatrix> explicitSource(
    volScalarField& progress,
    const volScalarField& reactionRate);

//- Convert a specific progress rate [1/time] to a volumetric mass source
//  using the solver-owned density, then assemble it with the same sign and
//  cell-volume convention as explicitSource().
tmp<fvScalarMatrix> explicitSpecificSource(
    volScalarField& progress,
    const volScalarField& reactionRate,
    const volScalarField& density);

}  // namespace Foam::foamNordic::combustion

#endif
