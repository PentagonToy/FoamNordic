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
#ifndef Foam_LESModels_kEqnFjord_H
#define Foam_LESModels_kEqnFjord_H

#include "LESeddyViscosity.H"

#include "closureHook.H"
#include "fieldProgramObservation.H"
#include "fieldProgramSession.H"
#include "operations/expression.H"

#include <cstdint>
#include <memory>
#include <string>

namespace Foam::LESModels {

template<class BasicTurbulenceModel>
class kEqnFjord final : public LESeddyViscosity<BasicTurbulenceModel> {
    volScalarField k_;
    dimensionedScalar Ck_;
    volScalarField kProduction_;
    volScalarField kDissipationCoeff_;
    std::unique_ptr<foamNordic::ClosureHook> closure_;
    bool nativeTransport_;
    bool directState_;
    foamNordic::operations::Expression nativeDiffusivity_;
    foamNordic::operations::Expression nativeSource_;
    foamNordic::operations::Expression nativeImplicitSink_;
    foamNordic::operations::Expression nativeNut_;
    scalar nativeLowerBound_;
    std::unique_ptr<foamNordic::FieldProgramObservation> nativeObservation_;
    foamNordic::MeshTopologySequence nativeTopology_;
    word nativeProfileProgram_;
    std::uint64_t nativeProfileEvery_;
    std::uint64_t nativeProfileSequence_;
    std::uint64_t nativeProfileSamples_;
    std::uint64_t nativePreSolveNanoseconds_;
    std::uint64_t nativeAssembleNanoseconds_;
    std::uint64_t nativeSolveNanoseconds_;
    std::uint64_t nativePostSolveNanoseconds_;
    std::uint64_t nativeCommitNanoseconds_;
    std::uint64_t nativeInvocationIndex_;

    kEqnFjord(const kEqnFjord&) = delete;
    void operator=(const kEqnFjord&) = delete;

    void updateClosure(const volTensorField& gradU);
    void updateState(const volTensorField& gradU);
    void validateNative(const volTensorField& gradU);
    void writeNativeTelemetry() const;

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

    ~kEqnFjord() override;

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
