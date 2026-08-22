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

#include "reactionRateFjord.H"

#include "fvmSup.H"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace Foam::combustionModels {

template<class ReactionThermo>
void reactionRateFjord<ReactionThermo>::validateContract() const {
    const dictionary& closure =
        this->coeffs().subDict("foamNordicClosure");
    const wordList inputs = closure.get<wordList>("inputs");
    const wordList outputs = closure.get<wordList>("outputs");

    std::unordered_set<std::string> inputNames;
    for (const auto& input : inputs) {
        inputNames.emplace(input.c_str());
    }
    for (const char* required : {"progress", "variance", "temperature"}) {
        if (!inputNames.contains(required)) {
            throw std::invalid_argument(
                std::string("reactionRateFjord requires logical input '")
                + required + "'.");
        }
    }

    if (outputs.size() != 1 || outputs[0] != reactionRateField_) {
        throw std::invalid_argument(
            "reactionRateFjord requires exactly one output matching "
            "reactionRateField.");
    }
    if (!this->mesh().template foundObject<volScalarField>(
            reactionRateField_)) {
        throw std::invalid_argument(
            "reactionRateFjord output must be a solver-owned volScalarField: "
            + std::string(reactionRateField_.c_str()));
    }

    const auto& reactionRate =
        this->mesh().template lookupObject<volScalarField>(reactionRateField_);
    if (reactionRate.dimensions() != reactionRateDimensions_) {
        throw std::invalid_argument(
            "reactionRateFjord output dimensions do not match "
            "reactionRateDimensions.");
    }
}

template<class ReactionThermo>
reactionRateFjord<ReactionThermo>::reactionRateFjord(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turbulence,
    const word& combustionProperties)
    : ThermoCombustion<ReactionThermo>(modelType, thermo, turbulence),
      reactionRateField_(this->coeffs().template get<word>(
          "reactionRateField")),
      reactionRateDimensions_(this->coeffs().template get<dimensionSet>(
          "reactionRateDimensions")),
      closure_(nullptr) {
    static_cast<void>(combustionProperties);
    validateContract();
    closure_ = std::make_unique<foamNordic::ClosureHook>(
        this->coeffs().subDict("foamNordicClosure"));
}

template<class ReactionThermo>
void reactionRateFjord<ReactionThermo>::correct() {
    if (!this->active()) {
        return;
    }
    static_cast<void>(closure_->invoke(this->mesh(), this->mesh().time()));
}

template<class ReactionThermo>
tmp<fvScalarMatrix> reactionRateFjord<ReactionThermo>::R(
    volScalarField& species) const {
    return tmp<fvScalarMatrix>::New(species, dimMass / dimTime);
}

template<class ReactionThermo>
tmp<volScalarField> reactionRateFjord<ReactionThermo>::Qdot() const {
    return volScalarField::New(
        this->thermo().phaseScopedName(typeName, "Qdot"),
        IOobject::NO_REGISTER,
        this->mesh(),
        dimensionedScalar(dimEnergy / dimVolume / dimTime, Zero));
}

template<class ReactionThermo>
bool reactionRateFjord<ReactionThermo>::read() {
    if (!ThermoCombustion<ReactionThermo>::read()) {
        return false;
    }

    const word configuredField =
        this->coeffs().template get<word>("reactionRateField");
    const dimensionSet configuredDimensions =
        this->coeffs().template get<dimensionSet>("reactionRateDimensions");
    if (configuredField != reactionRateField_
        || configuredDimensions != reactionRateDimensions_) {
        throw std::invalid_argument(
            "reactionRateFjord field identity and dimensions are immutable "
            "during a run.");
    }
    validateContract();
    return true;
}

}  // namespace Foam::combustionModels
