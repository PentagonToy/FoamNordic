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
#include "nativeAlgebraMonitor.H"
#include "foamCompatibility.H"

#include <stdexcept>

namespace Foam
{
namespace foamNordic
{

NativeAlgebraMonitor::NativeAlgebraMonitor(
    const dictionary& dict
) :
    observation_(FieldProgramObservation::create(dict)),
    program_(dictionaryValue<word>(dict, "profileProgram", "algebra"))
{
    const auto profileEvery = dictionaryValue<label>(dict, "profileEvery", 0);
    if (profileEvery < 0)
    {
        throw std::invalid_argument(
            "FoamNordic profileEvery must not be negative."
        );
    }
    profileEvery_ = static_cast<std::uint64_t>(profileEvery);
}

NativeAlgebraMonitor::~NativeAlgebraMonitor() noexcept
{
    shutdown();
}

void
NativeAlgebraMonitor::record(
    const fvMesh& mesh,
    const Time& time,
    const std::uint64_t exchangeIndex,
    const std::uint64_t elapsedNanoseconds
)
{
    if (stopped_)
    {
        return;
    }
    if (profileEvery_ != 0 && profileSequence_++ % profileEvery_ == 0)
    {
        ++profileSamples_;
        evaluateAndCommitNanoseconds_ += elapsedNanoseconds;
    }
    if (observation_)
    {
        observation_->publish(
            mesh,
            time,
            exchangeIndex,
            topology_.observe(mesh),
            static_cast<double>(elapsedNanoseconds) / 1.0e9
        );
    }
}

void
NativeAlgebraMonitor::shutdown() noexcept
{
    if (stopped_)
    {
        return;
    }
    stopped_ = true;
    if (observation_)
    {
        observation_->shutdown();
        observation_.reset();
    }
    writeNativeAlgebraTelemetry(
        program_,
        profileSamples_,
        evaluateAndCommitNanoseconds_
    );
}

} // End namespace foamNordic
} // End namespace Foam
