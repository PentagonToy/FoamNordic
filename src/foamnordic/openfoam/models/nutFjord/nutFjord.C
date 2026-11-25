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
#include "foamCompatibility.H"

#include "operations/frame.H"
#ifdef FOAMNORDIC_FOUNDATION
#include "fvConstraints.H"
#endif

#include <chrono>

namespace Foam
{
namespace LESModels
{

template<class BasicTurbulenceModel>
void
nutFjord<BasicTurbulenceModel>::correctNut()
{
    if (nativeAlgebra_)
    {
        const auto started = std::chrono::steady_clock::now();
        tmp<volScalarField> molecularViscosity = this->nu();
        foamNordic::operations::OperationFrame frame(this->mesh_);
        frame.bind("delta", this->delta());
        frame.bind("nu", molecularViscosity());
        this->nut_ = frame.scalar(nativeNut_);
        this->nut_.correctBoundaryConditions();
#ifdef FOAMNORDIC_FOUNDATION
        fvConstraints::New(this->mesh_).constrain(this->nut_);
#else
        BasicTurbulenceModel::correctNut();
#endif
        const auto elapsed
            = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started
            );
        nativeMonitor_->record(
            this->mesh_,
            this->runTime_,
            nativeInvocationIndex_++,
            static_cast<std::uint64_t>(elapsed.count())
        );
        return;
    }

    static_cast<void>(closure_->invoke(
        this->mesh_,
        this->runTime_,
        [this](foamNordic::operations::OperationFrame& frame)
        {
            frame.bind("delta", this->delta());
        }
    ));

    this->nut_.correctBoundaryConditions();
#ifdef FOAMNORDIC_FOUNDATION
    fvConstraints::New(this->mesh_).constrain(this->nut_);
#else
    BasicTurbulenceModel::correctNut();
#endif
}

template<class BasicTurbulenceModel>
nutFjord<BasicTurbulenceModel>::nutFjord(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& velocity,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
#ifndef FOAMNORDIC_FOUNDATION
    const word& propertiesName,
#endif
    const word& type
) :
    LESeddyViscosity<BasicTurbulenceModel>(
        type,
        alpha,
        rho,
        velocity,
        alphaRhoPhi,
        phi,
        transport
#ifndef FOAMNORDIC_FOUNDATION
        , propertiesName
#endif
    ),
    closure_(nullptr),
    nativeAlgebra_(false),
    nativeMonitor_(nullptr)
{
#ifdef FOAMNORDIC_FOUNDATION
    const dictionary& modelCoefficients = this->typeDict(type);
#else
    const dictionary& modelCoefficients = this->coeffDict_;
#endif
    if (modelCoefficients.found("foamNordicAlgebra"))
    {
        const dictionary& algebra = modelCoefficients.subDict("foamNordicAlgebra");
        nativeAlgebra_ = true;
        const dictionary& outputs = algebra.subDict("outputs");
        if (outputs.size() != 1 || !outputs.found("nut"))
        {
            FatalIOErrorInFunction(outputs)
                << "nutFjord native algebra requires exactly one nut output."
                << exit(FatalIOError);
        }
        nativeNut_ = foamNordic::operations::Expression::parse(
            Foam::foamNordic::dictionaryValue<Foam::string>(outputs, "nut")
        );
        nativeMonitor_
            = std::make_unique<foamNordic::NativeAlgebraMonitor>(algebra);
    }
    else
    {
        closure_ = std::make_unique<foamNordic::ClosureHook>(
            modelCoefficients.subDict("foamNordicClosure")
        );
    }
#ifndef FOAMNORDIC_FOUNDATION
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
#endif
}

template<class BasicTurbulenceModel> nutFjord<BasicTurbulenceModel>::~nutFjord()
{
    if (nativeMonitor_)
    {
        nativeMonitor_->shutdown();
    }
}

template<class BasicTurbulenceModel>
bool
nutFjord<BasicTurbulenceModel>::read()
{
    return LESeddyViscosity<BasicTurbulenceModel>::read();
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
nutFjord<BasicTurbulenceModel>::k() const
{
    return Foam::foamNordic::makeTemporary<volScalarField>(
        IOobject(
            IOobject::groupName("k", this->U_.group()),
            Foam::foamNordic::currentTimeName(this->runTime_),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
#ifdef FOAMNORDIC_FOUNDATION
            false
#else
            IOobject::NO_REGISTER
#endif
        ),
        this->mesh_,
        dimensionedScalar("zero", sqr(dimVelocity), 0)
    );
}

template<class BasicTurbulenceModel>
void
nutFjord<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
    LESeddyViscosity<BasicTurbulenceModel>::correct();
    correctNut();
}

} // End namespace LESModels
} // End namespace Foam
