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
#include "nutFjord.H"

#include "operations/frame.H"

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
void nutFjord<BasicTurbulenceModel>::correctNut() {
    if (nativeAlgebra_) {
        tmp<volScalarField> molecularViscosity = this->nu();
        foamNordic::operations::OperationFrame frame(this->mesh_);
        frame.bind("delta", this->delta());
        frame.bind("nu", molecularViscosity());
        this->nut_ = frame.scalar(nativeNut_);
    } else {
        static_cast<void>(closure_->invoke(
            this->mesh_,
            this->runTime_,
            [this](foamNordic::operations::OperationFrame& frame) {
                frame.bind("delta", this->delta());
            }));
    }

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
      closure_(nullptr),
      nativeAlgebra_(false) {
    const dictionary* algebra =
        this->coeffDict_.findDict("foamNordicAlgebra");
    if (algebra != nullptr) {
        nativeAlgebra_ = true;
        const dictionary& outputs = algebra->subDict("outputs");
        if (outputs.size() != 1 || !outputs.found("nut")) {
            FatalIOErrorInFunction(outputs)
                << "nutFjord native algebra requires exactly one nut output."
                << exit(FatalIOError);
        }
        nativeNut_ = foamNordic::operations::Expression::parse(
            outputs.get<Foam::string>("nut"));
    } else {
        closure_ = std::make_unique<foamNordic::ClosureHook>(
            this->coeffDict_.subDict("foamNordicClosure"));
    }
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
