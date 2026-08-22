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

#ifndef Foam_LESModels_nutFjord_H
#define Foam_LESModels_nutFjord_H

#include "LESeddyViscosity.H"

#include "closureHook.H"

#include <memory>

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
class nutFjord final : public LESeddyViscosity<BasicTurbulenceModel> {
    std::unique_ptr<foamNordic::ClosureHook> closure_;

    nutFjord(const nutFjord&) = delete;
    void operator=(const nutFjord&) = delete;

protected:
    void correctNut() override;

public:
    using alphaField = typename BasicTurbulenceModel::alphaField;
    using rhoField = typename BasicTurbulenceModel::rhoField;
    using transportModel = typename BasicTurbulenceModel::transportModel;

    TypeName("nutFjord");

    nutFjord(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& velocity,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    ~nutFjord() override = default;

    bool read() override;
    tmp<volScalarField> k() const override;
    void correct() override;
};

}  // namespace Foam::LESModels

#ifdef NoRepository
#include "nutFjord.C"
#endif

#endif
