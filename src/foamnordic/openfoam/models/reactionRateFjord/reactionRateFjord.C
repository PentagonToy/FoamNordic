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
#include "progressVariableSource.H"

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
    if (reactionRate_ == nullptr) {
        throw std::invalid_argument(
            "reactionRateFjord output field is unavailable: "
            + std::string(reactionRateField_.c_str()));
    }

    if (reactionRate_->dimensions() != reactionRateDimensions_) {
        throw std::invalid_argument(
            "reactionRateFjord output dimensions do not match "
            "reactionRateDimensions.");
    }
    const dimensionSet expectedDimensions =
        reactionRateBasis_ == "volumetricMass"
        ? dimMass / dimVolume / dimTime
        : dimless / dimTime;
    if (reactionRateBasis_ != "volumetricMass"
        && reactionRateBasis_ != "specific") {
        throw std::invalid_argument(
            "reactionRateFjord reactionRateBasis must be 'volumetricMass' "
            "or 'specific'.");
    }
    if (reactionRateDimensions_ != expectedDimensions) {
        throw std::invalid_argument(
            "reactionRateFjord reactionRateDimensions do not match "
            "reactionRateBasis.");
    }
    if (!this->mesh().template foundObject<volScalarField>(progressField_)) {
        throw std::invalid_argument(
            "reactionRateFjord progressField must be a solver-owned "
            "volScalarField: " + std::string(progressField_.c_str()));
    }
}

template<class ReactionThermo>
reactionRateFjord<ReactionThermo>::reactionRateFjord(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turbulence,
    const word& combustionProperties)
    : ThermoCombustion<ReactionThermo>(modelType, thermo, turbulence),
      progressField_(this->coeffs().template get<word>("progressField")),
      reactionRateField_(this->coeffs().template get<word>(
          "reactionRateField")),
      reactionRateBasis_(this->coeffs().template getOrDefault<word>(
          "reactionRateBasis", "volumetricMass")),
      reactionRateDimensions_(this->coeffs().template get<dimensionSet>(
          "reactionRateDimensions")),
      autoCreateReactionRate_(this->coeffs().template lookupOrDefault<bool>(
          "autoCreateReactionRate", false)),
      ownedReactionRate_(nullptr),
      reactionRate_(nullptr),
      closure_(nullptr) {
    static_cast<void>(combustionProperties);
    if (this->mesh().template foundObject<volScalarField>(
            reactionRateField_)) {
        reactionRate_ = &const_cast<fvMesh&>(this->mesh())
            .template lookupObjectRef<volScalarField>(reactionRateField_);
    } else if (autoCreateReactionRate_) {
        ownedReactionRate_ = std::make_unique<volScalarField>
        (
            IOobject
            (
                reactionRateField_,
                this->mesh().time().timeName(),
                this->mesh(),
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            this->mesh(),
            dimensionedScalar
            (
                reactionRateField_,
                reactionRateDimensions_,
                Zero
            )
        );
        reactionRate_ = ownedReactionRate_.get();
    }
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
    volScalarField& field) const {
    if (field.name() == progressField_) {
        if (reactionRateBasis_ == "specific") {
            const tmp<volScalarField> density = this->thermo().rho();
            return foamNordic::combustion::explicitSpecificSource(
                field, *reactionRate_, density());
        }
        return foamNordic::combustion::explicitSource(field, *reactionRate_);
    }
    return tmp<fvScalarMatrix>::New(field, dimMass / dimTime);
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
    const word configuredProgress =
        this->coeffs().template get<word>("progressField");
    const dimensionSet configuredDimensions =
        this->coeffs().template get<dimensionSet>("reactionRateDimensions");
    const word configuredBasis = this->coeffs().template getOrDefault<word>(
        "reactionRateBasis", "volumetricMass");
    const bool configuredAutoCreate =
        this->coeffs().template lookupOrDefault<bool>(
            "autoCreateReactionRate", false);
    if (configuredProgress != progressField_
        || configuredField != reactionRateField_
        || configuredBasis != reactionRateBasis_
        || configuredDimensions != reactionRateDimensions_
        || configuredAutoCreate != autoCreateReactionRate_) {
        throw std::invalid_argument(
            "reactionRateFjord field identity, basis, and dimensions are immutable "
            "during a run.");
    }
    validateContract();
    return true;
}

}  // namespace Foam::combustionModels
