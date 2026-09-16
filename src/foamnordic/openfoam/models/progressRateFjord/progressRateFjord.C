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
#include "progressRateFjord.H"

#include "operations/frame.H"

#include <stdexcept>

namespace Foam
{
namespace foamNordic
{

const dictionary&
progressRateFjord::validateContract(
    const dictionary& dictionary,
    const volScalarField& reactionRate
)
{
    const auto outputs = dictionary.get<wordList>("outputs");
    const auto keys
        = dictionary.getOrDefault<wordList>("outputKeys", wordList());
    if (outputs.size() != 1 || (!keys.empty() && keys.size() != 1))
    {
        throw std::invalid_argument(
            "progressRateFjord requires exactly one omega_c output mapped "
            "to the solver-owned reaction-rate field."
        );
    }
    const word key = keys.empty() ? outputs.front() : keys.front();
    if (outputs.front() != reactionRate.name() || key != "omega_c")
    {
        throw std::invalid_argument(
            "progressRateFjord requires exactly one omega_c output mapped "
            "to the solver-owned reaction-rate field."
        );
    }
    return dictionary;
}

progressRateFjord::progressRateFjord(
    const dictionary& dictionary,
    const volScalarField& progress,
    const volScalarField& variance,
    const volScalarField& temperature,
    volScalarField& reactionRate
) :
    progress_(progress),
    variance_(variance),
    temperature_(temperature),
    reactionRate_(reactionRate),
    normalizedVariance_(
        IOobject(
            "foamNordicNormalizedVariance",
            progress.time().timeName(),
            progress.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        progress.mesh(),
        dimensionedScalar("zero", dimless, Zero)
    ),
    closure_(validateContract(dictionary, reactionRate))
{
}

void
progressRateFjord::correct()
{
    forAll(normalizedVariance_, celli)
    {
        const scalar progress
            = min(max(progress_[celli], scalar(0)), scalar(1));
        const scalar maximumVariance = progress * (1 - progress);
        normalizedVariance_[celli] = maximumVariance > SMALL
            ? min(max(variance_[celli] / maximumVariance, scalar(0)), scalar(1))
            : 0;
    }
    forAll(normalizedVariance_.boundaryField(), patchi)
    {
        normalizedVariance_.boundaryFieldRef()[patchi]
            = normalizedVariance_.boundaryField()[patchi].patchInternalField();
    }

    static_cast<void>(closure_.invoke(
        progress_.mesh(),
        progress_.time(),
        [this](operations::OperationFrame& frame)
        {
            frame.bind("progress", progress_);
            frame.bind("variance", variance_);
            frame.bind("temperature", temperature_);
            frame.bind("normalized_variance", normalizedVariance_);
        }
    ));
}

void
progressRateFjord::shutdown()
{
    closure_.shutdown();
}

} // End namespace foamNordic
} // End namespace Foam
