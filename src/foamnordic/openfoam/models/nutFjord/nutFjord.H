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
#ifndef Foam_LESModels_nutFjord_H
#define Foam_LESModels_nutFjord_H

#include "LESeddyViscosity.H"

#include "closureHook.H"
#include "nativeAlgebraMonitor.H"
#include "operations/expression.H"

#include <cstdint>
#include <memory>

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                           Class nutFjord Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class nutFjord final : public LESeddyViscosity<BasicTurbulenceModel>
{
    // Private Data

    std::unique_ptr<foamNordic::ClosureHook> closure_;
    bool nativeAlgebra_;
    foamNordic::operations::Expression nativeNut_;
    std::unique_ptr<foamNordic::NativeAlgebraMonitor> nativeMonitor_;
    std::uint64_t nativeInvocationIndex_ { 0 };


    // Private Member Functions

    nutFjord(const nutFjord&) = delete;
    void operator=(const nutFjord&) = delete;

protected:
    // Protected Member Functions

    //- Update the subgrid-scale eddy viscosity
    void correctNut() override;


public:
    // Public Types

    using alphaField = typename BasicTurbulenceModel::alphaField;
    using rhoField = typename BasicTurbulenceModel::rhoField;
#ifdef FOAMNORDIC_FOUNDATION
    using transportModel = Foam::viscosity;
#else
    using transportModel = typename BasicTurbulenceModel::transportModel;
#endif

    TypeName("nutFjord");


    // Constructors

    nutFjord(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& velocity,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
#ifndef FOAMNORDIC_FOUNDATION
        const word& propertiesName = turbulenceModel::propertiesName,
#endif
        const word& type = typeName
    );

    //- Release the closure and observation resources
    ~nutFjord() override;


    // Member Functions

    bool read() override;
    tmp<volScalarField> k() const override;
    void correct() override;
};

} // End namespace LESModels
} // End namespace Foam

#ifdef NoRepository
#include "nutFjord.C"
#endif

#endif
