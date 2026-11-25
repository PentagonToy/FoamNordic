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
#include "kEqnFjord.H"
#include "foamCompatibility.H"

#include "Pstream.H"
#include "bound.H"
#ifdef FOAMNORDIC_FOUNDATION
#include "fvConstraints.H"
#include "fvModels.H"
#else
#include "fvOptions.H"
#endif
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "operations/frame.H"

#include <chrono>
#include <stdexcept>

namespace Foam
{
namespace LESModels
{

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::updateClosure(
    const volTensorField& gradU
)
{
    this->nut_ = Ck_ * sqrt(k_) * this->delta();
#ifdef FOAMNORDIC_FOUNDATION
    kProduction_ = this->nut_ * (gradU && dev(twoSymm(gradU)));
#else
    kProduction_ = this->nut_ * (gradU && devTwoSymm(gradU));
#endif
    kDissipationCoeff_ = this->Ce_ * sqrt(k_) / this->delta();

    static_cast<void>(closure_->invoke(
        this->mesh_,
        this->runTime_,
        [this, &gradU](foamNordic::operations::OperationFrame& frame)
        {
            frame.bind("k", k_);
            frame.bind("grad(U)", gradU);
            frame.bind("delta", this->delta());
        }
    ));
    correctNut();
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::updateState(
    const volTensorField& gradU
)
{
    tmp<volScalarField> molecularViscosity = this->nu();
    static_cast<void>(closure_->invoke(
        this->mesh_,
        this->runTime_,
        [this,
         &gradU,
         &molecularViscosity](foamNordic::operations::OperationFrame& frame)
        {
            frame.bind("k", k_);
            frame.bind("grad(U)", gradU);
            frame.bind("delta", this->delta());
            frame.bind("nu", molecularViscosity());
        }
    ));
    bound(k_, this->kMin_);
    correctNut();
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::validateNative(
    const volTensorField& gradU
)
{
    tmp<volScalarField> molecularViscosity = this->nu();
    foamNordic::operations::OperationFrame frame(this->mesh_);
    frame.bind("k", k_);
    frame.bind("grad(U)", gradU);
    frame.bind("delta", this->delta());
    frame.bind("nu", molecularViscosity());
    static_cast<void>(frame.scalar(nativeDiffusivity_));
    static_cast<void>(frame.scalar(nativeSource_));
    static_cast<void>(frame.scalar(nativeImplicitSink_));
    this->nut_ = frame.scalar(nativeNut_);
    correctNut();
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::writeNativeTelemetry() const
{
    if (nativeProfileSamples_ == 0)
    {
        return;
    }
    const auto mean = [this](std::uint64_t total)
    {
        return total / nativeProfileSamples_;
    };
    Info << "[FoamNordic] Native transport telemetry: {\"program\":\""
         << nativeProfileProgram_ << "\",\"rank\":" << Pstream::myProcNo()
         << ",\"samples\":" << nativeProfileSamples_
         << ",\"mean_pre_solve_nanoseconds\":"
         << mean(nativePreSolveNanoseconds_)
         << ",\"mean_assemble_nanoseconds\":"
         << mean(nativeAssembleNanoseconds_)
         << ",\"mean_solve_nanoseconds\":" << mean(nativeSolveNanoseconds_)
         << ",\"mean_post_solve_nanoseconds\":"
         << mean(nativePostSolveNanoseconds_)
         << ",\"mean_commit_nanoseconds\":" << mean(nativeCommitNanoseconds_)
         << '}' << nl;
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::correctNut()
{
    this->nut_.correctBoundaryConditions();
#ifdef FOAMNORDIC_FOUNDATION
    fvConstraints::New(this->mesh_).constrain(this->nut_);
#else
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();
#endif
}

template<class BasicTurbulenceModel>
tmp<fvScalarMatrix>
kEqnFjord<BasicTurbulenceModel>::kSource() const
{
    return Foam::foamNordic::makeTemporary<fvScalarMatrix>(
        k_,
        dimVolume * this->rho_.dimensions() * k_.dimensions() / dimTime
    );
}

template<class BasicTurbulenceModel>
kEqnFjord<BasicTurbulenceModel>::kEqnFjord(
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
    k_(IOobject(
           IOobject::groupName("k", this->alphaRhoPhi_.group()),
           Foam::foamNordic::currentTimeName(this->runTime_),
           this->mesh_,
           IOobject::MUST_READ,
           IOobject::AUTO_WRITE
       ),
       this->mesh_),
#ifdef FOAMNORDIC_FOUNDATION
    Ck_("Ck", this->typeDict(type), 0.094),
#else
    Ck_(dimensioned<scalar>::getOrAddToDict("Ck", this->coeffDict_, 0.094)),
#endif
    kProduction_(
        IOobject(
            IOobject::groupName("kProduction", this->alphaRhoPhi_.group()),
            Foam::foamNordic::currentTimeName(this->runTime_),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("zero", sqr(dimVelocity) / dimTime, 0)
    ),
    kDissipationCoeff_(
        IOobject(
            IOobject::groupName(
                "kDissipationCoeff",
                this->alphaRhoPhi_.group()
            ),
            Foam::foamNordic::currentTimeName(this->runTime_),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("zero", dimless / dimTime, 0)
    ),
    closure_(nullptr),
    nativeTransport_(false),
    directState_(false),
    nativeLowerBound_(0),
    nativeObservation_(nullptr),
    nativeProfileProgram_(typeName),
    nativeProfileEvery_(0),
    nativeProfileSequence_(0),
    nativeProfileSamples_(0),
    nativePreSolveNanoseconds_(0),
    nativeAssembleNanoseconds_(0),
    nativeSolveNanoseconds_(0),
    nativePostSolveNanoseconds_(0),
    nativeCommitNanoseconds_(0),
    nativeInvocationIndex_(0)
{
#ifdef FOAMNORDIC_FOUNDATION
    const dictionary& modelCoefficients = this->typeDict(type);
#else
    const dictionary& modelCoefficients = this->coeffDict_;
#endif
    nativeTransport_ = modelCoefficients.found("foamNordicTransport");
    if (nativeTransport_)
    {
        const dictionary& transport
            = modelCoefficients.subDict("foamNordicTransport");
        const word transportedField = Foam::foamNordic::dictionaryValue<word>(transport, "field");
        if (transportedField != "k")
        {
            FatalIOErrorInFunction(transport)
                << "kEqnFjord native transport must advance field k."
                << exit(FatalIOError);
        }
        nativeDiffusivity_ = foamNordic::operations::Expression::parse(
            Foam::foamNordic::dictionaryValue<Foam::string>(transport, "diffusivity")
        );
        nativeSource_ = foamNordic::operations::Expression::parse(
            Foam::foamNordic::dictionaryValue<Foam::string>(transport, "source")
        );
        nativeImplicitSink_ = foamNordic::operations::Expression::parse(
            Foam::foamNordic::dictionaryValue<Foam::string>(transport, "implicitSink")
        );
        nativeNut_ = foamNordic::operations::Expression::parse(
            Foam::foamNordic::dictionaryValue<Foam::string>(transport, "nut")
        );
        nativeLowerBound_ = Foam::foamNordic::dictionaryValue<scalar>(transport, "lowerBound");
        const auto configuredProfileEvery
            = Foam::foamNordic::dictionaryValue<label>(transport, "profileEvery", 0);
        if (configuredProfileEvery < 0)
        {
            throw std::invalid_argument(
                "FoamNordic profileEvery must not be negative."
            );
        }
        nativeProfileEvery_
            = static_cast<std::uint64_t>(configuredProfileEvery);
        nativeProfileProgram_
            = Foam::foamNordic::dictionaryValue<word>(transport, "profileProgram", typeName);
        nativeObservation_
            = foamNordic::FieldProgramObservation::create(transport);
    }
    else
    {
        const dictionary& closureDict
            = modelCoefficients.subDict("foamNordicClosure");
        const wordList outputFields = Foam::foamNordic::dictionaryValue<wordList>(closureDict, "outputs");
        directState_ = outputFields.size() == 2
            && ((outputFields[0] == "k" && outputFields[1] == "nut")
                || (outputFields[0] == "nut" && outputFields[1] == "k"));
        closure_ = std::make_unique<foamNordic::ClosureHook>(closureDict);
    }
    bound(k_, this->kMin_);
#ifndef FOAMNORDIC_FOUNDATION
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
#endif
}

template<class BasicTurbulenceModel>
kEqnFjord<BasicTurbulenceModel>::~kEqnFjord()
{
    if (nativeObservation_)
    {
        nativeObservation_->shutdown();
    }
    writeNativeTelemetry();
}

template<class BasicTurbulenceModel>
bool
kEqnFjord<BasicTurbulenceModel>::read()
{
    if (!LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        return false;
    }
#ifdef FOAMNORDIC_FOUNDATION
    Ck_.readIfPresent(this->typeDict());
#else
    Ck_.readIfPresent(this->coeffDict());
#endif
    return true;
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::validate()
{
    const tmp<volTensorField> gradU = fvc::grad(this->U_);
    if (nativeTransport_)
    {
        validateNative(gradU());
    }
    else if (directState_)
    {
        updateState(gradU());
    }
    else
    {
        updateClosure(gradU());
    }
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
kEqnFjord<BasicTurbulenceModel>::DkEff() const
{
    return Foam::foamNordic::makeTemporary<volScalarField>("DkEff", this->nut_ + this->nu());
}

template<class BasicTurbulenceModel>
void
kEqnFjord<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& velocity = this->U_;
#ifdef FOAMNORDIC_FOUNDATION
    const auto& models = fvModels::New(this->mesh_);
    const auto& constraints = fvConstraints::New(this->mesh_);
#else
    auto& options = fv::options::New(this->mesh_);
#endif
    const auto solveK = [&](tmp<fvScalarMatrix>& equation)
    {
        equation.ref().relax();
#ifdef FOAMNORDIC_FOUNDATION
        constraints.constrain(equation.ref());
        solve(equation);
        constraints.constrain(k_);
#else
        options.constrain(equation.ref());
        solve(equation);
        options.correct(k_);
#endif
    };

    LESeddyViscosity<BasicTurbulenceModel>::correct();
    tmp<volTensorField> gradU = fvc::grad(velocity);

    if (directState_)
    {
        updateState(gradU());
        return;
    }

    const volScalarField divU(fvc::div(fvc::absolute(this->phi(), velocity)));

    if (nativeTransport_)
    {
        using Clock = std::chrono::steady_clock;
        const bool sampled = nativeProfileEvery_ != 0
            && nativeProfileSequence_++ % nativeProfileEvery_ == 0;
        auto phaseStarted = sampled ? Clock::now() : Clock::time_point {};
        const auto accumulate = [sampled, &phaseStarted](std::uint64_t& total)
        {
            if (!sampled)
            {
                return;
            }
            const auto now = Clock::now();
            total += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now - phaseStarted
                )
                    .count()
            );
            phaseStarted = now;
        };

        tmp<volScalarField> molecularViscosity = this->nu();
        foamNordic::operations::OperationFrame frame(this->mesh_);
        frame.bind("k", k_);
        frame.bind("grad(U)", gradU());
        frame.bind("delta", this->delta());
        frame.bind("nu", molecularViscosity());
        const volScalarField& diffusivity = frame.scalar(nativeDiffusivity_);
        const volScalarField& production = frame.scalar(nativeSource_);
        const volScalarField& dissipation = frame.scalar(nativeImplicitSink_);
        accumulate(nativePreSolveNanoseconds_);

        tmp<fvScalarMatrix> equation(
            fvm::ddt(alpha, rho, k_) + fvm::div(alphaRhoPhi, k_)
                - fvm::laplacian(alpha * rho * diffusivity, k_)
            == alpha * rho * production
                - fvm::SuSp((2.0 / 3.0) * alpha * rho * divU, k_)
                - fvm::Sp(alpha * rho * dissipation, k_) + kSource()
#ifdef FOAMNORDIC_FOUNDATION
                + models.source(alpha, rho, k_)
#else
                + options(alpha, rho, k_)
#endif
        );
        accumulate(nativeAssembleNanoseconds_);

        solveK(equation);
        bound(
            k_,
            dimensionedScalar(
                "FoamNordic:kLowerBound",
                k_.dimensions(),
                nativeLowerBound_
            )
        );
        accumulate(nativeSolveNanoseconds_);

        foamNordic::operations::OperationFrame postSolve(this->mesh_);
        postSolve.bind("k", k_);
        postSolve.bind("grad(U)", gradU());
        postSolve.bind("delta", this->delta());
        postSolve.bind("nu", molecularViscosity());
        this->nut_ = postSolve.scalar(nativeNut_);
        accumulate(nativePostSolveNanoseconds_);
        correctNut();
        accumulate(nativeCommitNanoseconds_);
        if (sampled)
        {
            ++nativeProfileSamples_;
        }
        if (nativeObservation_)
        {
            nativeObservation_->publish(
                this->mesh_,
                this->runTime_,
                nativeInvocationIndex_,
                nativeTopology_.observe(this->mesh_),
                0.0
            );
        }
        ++nativeInvocationIndex_;
        return;
    }

    updateClosure(gradU());

    tmp<fvScalarMatrix> equation(
        fvm::ddt(alpha, rho, k_) + fvm::div(alphaRhoPhi, k_)
            - fvm::laplacian(alpha * rho * DkEff(), k_)
        == alpha * rho * kProduction_
            - fvm::SuSp((2.0 / 3.0) * alpha * rho * divU, k_)
            - fvm::Sp(alpha * rho * kDissipationCoeff_, k_) + kSource()
#ifdef FOAMNORDIC_FOUNDATION
            + models.source(alpha, rho, k_)
#else
            + options(alpha, rho, k_)
#endif
    );

    solveK(equation);
    bound(k_, this->kMin_);

    updateClosure(gradU());
}

} // End namespace LESModels
} // End namespace Foam
