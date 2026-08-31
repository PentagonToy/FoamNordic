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

#ifndef Foam_combustionModels_reactionRateFjord_H
#define Foam_combustionModels_reactionRateFjord_H

#include "ThermoCombustion.H"

#include "closureHook.H"

#include <memory>

namespace Foam::combustionModels {

template<class ReactionThermo>
class reactionRateFjord final : public ThermoCombustion<ReactionThermo> {
    word progressField_;
    word reactionRateField_;
    word reactionRateBasis_;
    dimensionSet reactionRateDimensions_;
    bool autoCreateReactionRate_;
    std::unique_ptr<volScalarField> ownedReactionRate_;
    volScalarField* reactionRate_;
    std::unique_ptr<foamNordic::ClosureHook> closure_;

    reactionRateFjord(const reactionRateFjord&) = delete;
    void operator=(const reactionRateFjord&) = delete;

    void validateContract() const;

public:
    TypeName("reactionRateFjord");

    reactionRateFjord(
        const word& modelType,
        ReactionThermo& thermo,
        const compressibleTurbulenceModel& turbulence,
        const word& combustionProperties);

    ~reactionRateFjord() override = default;

    void correct() override;
    tmp<fvScalarMatrix> R(volScalarField& species) const override;
    tmp<volScalarField> Qdot() const override;
    bool read() override;
};

}  // namespace Foam::combustionModels

#ifdef NoRepository
#include "reactionRateFjord.C"
#endif

#endif
