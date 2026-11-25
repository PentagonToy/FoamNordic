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
#ifdef FOAMNORDIC_FOUNDATION
#include "makeCompressibleMomentumTransportModel.H"
#else
#include "turbulentFluidThermoModels.H"
#endif

#include "nutFjord.H"
#include "kEqnFjord.H"

makeLESModel(nutFjord);
makeLESModel(kEqnFjord);
