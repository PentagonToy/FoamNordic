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

#include "kEqnFjord.H"

#include "bound.H"
#include "fvOptions.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "operations/frame.H"

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
void kEqnFjord<BasicTurbulenceModel>::updateClosure(
    const volTensorField& gradU) {
    this->nut_ = Ck_ * sqrt(k_) * this->delta();
    kProduction_ = this->nut_ * (gradU && devTwoSymm(gradU));
    kDissipationCoeff_ = this->Ce_ * sqrt(k_) / this->delta();

    static_cast<void>(closure_->invoke(
        this->mesh_,
        this->runTime_,
        [this, &gradU](foamNordic::operations::OperationFrame& frame) {
            frame.bind("k", k_);
            frame.bind("grad(U)", gradU);
            frame.bind("delta", this->delta());
        }));
    correctNut();
}

template<class BasicTurbulenceModel>
void kEqnFjord<BasicTurbulenceModel>::correctNut() {
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();
}

template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kEqnFjord<BasicTurbulenceModel>::kSource() const {
    return tmp<fvScalarMatrix>::New(
        k_,
        dimVolume * this->rho_.dimensions() * k_.dimensions() / dimTime);
}

template<class BasicTurbulenceModel>
kEqnFjord<BasicTurbulenceModel>::kEqnFjord(
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
      k_(
          IOobject(
              IOobject::groupName("k", this->alphaRhoPhi_.group()),
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      Ck_(dimensioned<scalar>::getOrAddToDict("Ck", this->coeffDict_, 0.094)),
      kProduction_(
          IOobject(
              IOobject::groupName(
                  "kProduction", this->alphaRhoPhi_.group()),
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("zero", sqr(dimVelocity) / dimTime, 0)),
      kDissipationCoeff_(
          IOobject(
              IOobject::groupName(
                  "kDissipationCoeff", this->alphaRhoPhi_.group()),
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("zero", dimless / dimTime, 0)),
      closure_(std::make_unique<foamNordic::ClosureHook>(
          this->coeffDict_.subDict("foamNordicClosure"))) {
    bound(k_, this->kMin_);
    if (type == typeName) {
        this->printCoeffs(type);
    }
}

template<class BasicTurbulenceModel>
bool kEqnFjord<BasicTurbulenceModel>::read() {
    if (!LESeddyViscosity<BasicTurbulenceModel>::read()) {
        return false;
    }
    Ck_.readIfPresent(this->coeffDict());
    return true;
}

template<class BasicTurbulenceModel>
void kEqnFjord<BasicTurbulenceModel>::validate() {
    const tmp<volTensorField> gradU = fvc::grad(this->U_);
    updateClosure(gradU());
}

template<class BasicTurbulenceModel>
tmp<volScalarField> kEqnFjord<BasicTurbulenceModel>::DkEff() const {
    return tmp<volScalarField>::New("DkEff", this->nut_ + this->nu());
}

template<class BasicTurbulenceModel>
void kEqnFjord<BasicTurbulenceModel>::correct() {
    if (!this->turbulence_) {
        return;
    }

    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& velocity = this->U_;
    fv::options& options = fv::options::New(this->mesh_);

    LESeddyViscosity<BasicTurbulenceModel>::correct();
    const volScalarField divU(fvc::div(fvc::absolute(this->phi(), velocity)));
    tmp<volTensorField> gradU = fvc::grad(velocity);

    updateClosure(gradU());

    tmp<fvScalarMatrix> equation(
        fvm::ddt(alpha, rho, k_)
        + fvm::div(alphaRhoPhi, k_)
        - fvm::laplacian(alpha * rho * DkEff(), k_)
        == alpha * rho * kProduction_
           - fvm::SuSp((2.0 / 3.0) * alpha * rho * divU, k_)
           - fvm::Sp(alpha * rho * kDissipationCoeff_, k_) + kSource()
           + options(alpha, rho, k_));

    equation.ref().relax();
    options.constrain(equation.ref());
    solve(equation);
    options.correct(k_);
    bound(k_, this->kMin_);

    updateClosure(gradU());
}

}  // namespace Foam::LESModels
