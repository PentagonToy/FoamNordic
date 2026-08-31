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

#ifndef Foam_LESModels_kEqnFjord_H
#define Foam_LESModels_kEqnFjord_H

#include "LESeddyViscosity.H"

#include "closureHook.H"

#include <memory>

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
class kEqnFjord final : public LESeddyViscosity<BasicTurbulenceModel> {
    volScalarField k_;
    dimensionedScalar Ck_;
    volScalarField kProduction_;
    volScalarField kDissipationCoeff_;
    std::unique_ptr<foamNordic::ClosureHook> closure_;

    kEqnFjord(const kEqnFjord&) = delete;
    void operator=(const kEqnFjord&) = delete;

    void updateClosure(const volTensorField& gradU);

protected:
    void correctNut() override;
    tmp<fvScalarMatrix> kSource() const;

public:
    using alphaField = typename BasicTurbulenceModel::alphaField;
    using rhoField = typename BasicTurbulenceModel::rhoField;
    using transportModel = typename BasicTurbulenceModel::transportModel;

    TypeName("kEqnFjord");

    kEqnFjord(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& velocity,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    ~kEqnFjord() override = default;

    bool read() override;
    void validate() override;
    tmp<volScalarField> k() const override { return k_; }
    tmp<volScalarField> DkEff() const;
    void correct() override;
};

}  // namespace Foam::LESModels

#ifdef NoRepository
#include "kEqnFjord.C"
#endif

#endif
