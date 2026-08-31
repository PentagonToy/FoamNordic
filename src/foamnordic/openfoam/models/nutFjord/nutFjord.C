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

#include "nutFjord.H"

#include "operations/frame.H"

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
void nutFjord<BasicTurbulenceModel>::correctNut() {
    static_cast<void>(closure_->invoke(
        this->mesh_,
        this->runTime_,
        [this](foamNordic::operations::OperationFrame& frame) {
            frame.bind("delta", this->delta());
        }));

    this->nut_.correctBoundaryConditions();
    BasicTurbulenceModel::correctNut();
}

template<class BasicTurbulenceModel>
nutFjord<BasicTurbulenceModel>::nutFjord(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& velocity,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type)
    : LESeddyViscosity<BasicTurbulenceModel>(
          type,
          alpha,
          rho,
          velocity,
          alphaRhoPhi,
          phi,
          transport,
          propertiesName),
      closure_(std::make_unique<foamNordic::ClosureHook>(
          this->coeffDict_.subDict("foamNordicClosure"))) {
    if (type == typeName) {
        this->printCoeffs(type);
    }
}

template<class BasicTurbulenceModel>
bool nutFjord<BasicTurbulenceModel>::read() {
    return LESeddyViscosity<BasicTurbulenceModel>::read();
}

template<class BasicTurbulenceModel>
tmp<volScalarField> nutFjord<BasicTurbulenceModel>::k() const {
    return tmp<volScalarField>::New(
        IOobject(
            IOobject::groupName("k", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER),
        this->mesh_,
        dimensionedScalar("zero", sqr(dimVelocity), 0));
}

template<class BasicTurbulenceModel>
void nutFjord<BasicTurbulenceModel>::correct() {
    if (!this->turbulence_) {
        return;
    }
    LESeddyViscosity<BasicTurbulenceModel>::correct();
    correctNut();
}

}  // namespace Foam::LESModels
