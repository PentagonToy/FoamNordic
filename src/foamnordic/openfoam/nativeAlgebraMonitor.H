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
#ifndef Foam_foamNordic_nativeAlgebraMonitor_H
#define Foam_foamNordic_nativeAlgebraMonitor_H

#include "fieldProgramObservation.H"
#include "fieldProgramSession.H"

#include "dictionary.H"
#include "fvMesh.H"

#include <cstdint>
#include <memory>

namespace Foam
{
namespace foamNordic
{

class NativeAlgebraMonitor final
{
    std::unique_ptr<FieldProgramObservation> observation_;
    MeshTopologySequence topology_;
    word program_;
    std::uint64_t profileEvery_ { 0 };
    std::uint64_t profileSequence_ { 0 };
    std::uint64_t profileSamples_ { 0 };
    std::uint64_t evaluateAndCommitNanoseconds_ { 0 };
    bool stopped_ { false };

public:
    explicit NativeAlgebraMonitor(const dictionary& dict);
    ~NativeAlgebraMonitor() noexcept;

    NativeAlgebraMonitor(const NativeAlgebraMonitor&) = delete;
    NativeAlgebraMonitor& operator=(const NativeAlgebraMonitor&) = delete;

    void record(
        const fvMesh& mesh,
        const Time& time,
        std::uint64_t exchangeIndex,
        std::uint64_t elapsedNanoseconds
    );
    void shutdown() noexcept;
};

} // End namespace foamNordic
} // End namespace Foam

#endif
