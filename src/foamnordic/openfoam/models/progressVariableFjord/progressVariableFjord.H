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

#ifndef Foam_combustionModels_progressVariableFjord_H
#define Foam_combustionModels_progressVariableFjord_H

#include "ThermoCombustion.H"

#include "closureHook.H"

#include <memory>

namespace Foam::combustionModels {

template<class ReactionThermo>
class progressVariableFjord final : public ThermoCombustion<ReactionThermo> {
    word progressField_;
    word reactionRateField_;
    dimensionSet reactionRateDimensions_;
    Switch correctThermo_;
    std::unique_ptr<foamNordic::ClosureHook> reactionRate_;
    std::unique_ptr<foamNordic::ClosureHook> manifold_;

    progressVariableFjord(const progressVariableFjord&) = delete;
    void operator=(const progressVariableFjord&) = delete;

    void validateContract() const;

public:
    TypeName("progressVariableFjord");

    progressVariableFjord(
        const word& modelType,
        ReactionThermo& thermo,
        const compressibleTurbulenceModel& turbulence,
        const word& combustionProperties);

    ~progressVariableFjord() override = default;

    void correct() override;
    tmp<fvScalarMatrix> R(volScalarField& species) const override;
    tmp<volScalarField> Qdot() const override;
    bool read() override;
};

}  // namespace Foam::combustionModels

#ifdef NoRepository
#include "progressVariableFjord.C"
#endif

#endif
