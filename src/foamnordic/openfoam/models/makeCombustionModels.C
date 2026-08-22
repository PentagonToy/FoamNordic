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

#include "makeCombustionTypes.H"

#include "psiReactionThermo.H"
#include "reactionRateFjord.H"
#include "rhoReactionThermo.H"

namespace Foam {

makeCombustionTypes(reactionRateFjord, psiReactionThermo);
makeCombustionTypes(reactionRateFjord, rhoReactionThermo);

}  // namespace Foam
