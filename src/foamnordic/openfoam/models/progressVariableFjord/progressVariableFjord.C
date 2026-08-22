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

#include "progressVariableFjord.H"

#include "fvmSup.H"
#include "progressVariableSource.H"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace Foam::combustionModels {

namespace {

void requireInputs(
    const dictionary& closure,
    const std::initializer_list<const char*> required,
    const char* label) {
    const wordList inputs = closure.get<wordList>("inputs");
    std::unordered_set<std::string> names;
    for (const auto& input : inputs) {
        names.emplace(input.c_str());
    }
    for (const char* name : required) {
        if (!names.contains(name)) {
            throw std::invalid_argument(
                std::string("progressVariableFjord ") + label
                + " requires logical input '" + name + "'.");
        }
    }
}

void requireScalarOutputs(
    const fvMesh& mesh,
    const dictionary& closure,
    const char* label) {
    const wordList outputs = closure.get<wordList>("outputs");
    if (outputs.empty()) {
        throw std::invalid_argument(
            std::string("progressVariableFjord ") + label
            + " requires at least one output.");
    }
    for (const auto& output : outputs) {
        if (!mesh.foundObject<volScalarField>(output)) {
            throw std::invalid_argument(
                std::string("progressVariableFjord ") + label
                + " output must be a solver-owned volScalarField: "
                + output.c_str());
        }
    }
}

}  // namespace

template<class ReactionThermo>
void progressVariableFjord<ReactionThermo>::validateContract() const {
    const word sourceTreatment = this->coeffs().template getOrDefault<word>(
        "sourceTreatment", "lagged");
    const word correctionStage = this->coeffs().template getOrDefault<word>(
        "correctionStage", "outerCorrector");
    if (sourceTreatment != "lagged") {
        throw std::invalid_argument(
            "progressVariableFjord currently supports sourceTreatment "
            "'lagged' only.");
    }
    if (correctionStage != "outerCorrector") {
        throw std::invalid_argument(
            "progressVariableFjord currently supports correctionStage "
            "'outerCorrector' only.");
    }

    const dictionary& reactionRate =
        this->coeffs().subDict("reactionRateClosure");
    const dictionary& manifold = this->coeffs().subDict("manifoldClosure");

    requireInputs(
        reactionRate,
        {"progress", "variance", "temperature"},
        "reaction-rate closure");
    requireInputs(manifold, {"progress", "variance"}, "manifold closure");
    requireScalarOutputs(this->mesh(), reactionRate, "reaction-rate closure");
    requireScalarOutputs(this->mesh(), manifold, "manifold closure");

    const wordList reactionOutputs = reactionRate.get<wordList>("outputs");
    if (reactionOutputs.size() != 1
        || reactionOutputs[0] != reactionRateField_) {
        throw std::invalid_argument(
            "progressVariableFjord requires one reaction-rate output matching "
            "reactionRateField.");
    }
    const auto& source =
        this->mesh().template lookupObject<volScalarField>(reactionRateField_);
    if (source.dimensions() != reactionRateDimensions_) {
        throw std::invalid_argument(
            "progressVariableFjord reaction-rate dimensions do not match "
            "reactionRateDimensions.");
    }
    if (!this->mesh().template foundObject<volScalarField>(progressField_)) {
        throw std::invalid_argument(
            "progressVariableFjord progressField must be a solver-owned "
            "volScalarField: " + std::string(progressField_.c_str()));
    }

    const wordList manifoldOutputs = manifold.get<wordList>("outputs");
    if (manifoldOutputs.found(reactionRateField_)) {
        throw std::invalid_argument(
            "progressVariableFjord manifold must not overwrite the "
            "reaction-rate source.");
    }
}

template<class ReactionThermo>
progressVariableFjord<ReactionThermo>::progressVariableFjord(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turbulence,
    const word& combustionProperties)
    : ThermoCombustion<ReactionThermo>(modelType, thermo, turbulence),
      progressField_(this->coeffs().template get<word>("progressField")),
      reactionRateField_(this->coeffs().template get<word>(
          "reactionRateField")),
      reactionRateDimensions_(this->coeffs().template get<dimensionSet>(
          "reactionRateDimensions")),
      correctThermo_(this->coeffs().template getOrDefault<Switch>(
          "correctThermo", true)),
      reactionRate_(nullptr),
      manifold_(nullptr) {
    static_cast<void>(combustionProperties);
    validateContract();
    reactionRate_ = std::make_unique<foamNordic::ClosureHook>(
        this->coeffs().subDict("reactionRateClosure"));
    manifold_ = std::make_unique<foamNordic::ClosureHook>(
        this->coeffs().subDict("manifoldClosure"));
}

template<class ReactionThermo>
void progressVariableFjord<ReactionThermo>::correct() {
    if (!this->active()) {
        return;
    }

    static_cast<void>(reactionRate_->invoke(this->mesh(), this->mesh().time()));
    static_cast<void>(manifold_->invoke(this->mesh(), this->mesh().time()));
    if (correctThermo_) {
        this->thermo().correct();
    }
}

template<class ReactionThermo>
tmp<fvScalarMatrix> progressVariableFjord<ReactionThermo>::R(
    volScalarField& field) const {
    if (field.name() == progressField_) {
        const auto& source = this->mesh().template lookupObject<volScalarField>(
            reactionRateField_);
        return foamNordic::combustion::explicitSource(field, source);
    }
    return tmp<fvScalarMatrix>::New(field, dimMass / dimTime);
}

template<class ReactionThermo>
tmp<volScalarField> progressVariableFjord<ReactionThermo>::Qdot() const {
    return volScalarField::New(
        this->thermo().phaseScopedName(typeName, "Qdot"),
        IOobject::NO_REGISTER,
        this->mesh(),
        dimensionedScalar(dimEnergy / dimVolume / dimTime, Zero));
}

template<class ReactionThermo>
bool progressVariableFjord<ReactionThermo>::read() {
    if (!ThermoCombustion<ReactionThermo>::read()) {
        return false;
    }

    const word configuredField =
        this->coeffs().template get<word>("reactionRateField");
    const word configuredProgress =
        this->coeffs().template get<word>("progressField");
    const dimensionSet configuredDimensions =
        this->coeffs().template get<dimensionSet>("reactionRateDimensions");
    if (configuredProgress != progressField_
        || configuredField != reactionRateField_
        || configuredDimensions != reactionRateDimensions_) {
        throw std::invalid_argument(
            "progressVariableFjord field identity and dimensions are "
            "immutable during a run.");
    }
    correctThermo_ = this->coeffs().template getOrDefault<Switch>(
        "correctThermo", true);
    validateContract();
    return true;
}

}  // namespace Foam::combustionModels
