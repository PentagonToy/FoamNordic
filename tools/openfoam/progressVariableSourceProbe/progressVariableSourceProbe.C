// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * FoamNordic
 */
// clang-format on

#include "fvCFD.H"

#include "progressVariableSource.H"

#include <cmath>
#include <stdexcept>

using namespace Foam;

int main(int argc, char* argv[]) {
    Foam::argList::noParallel();

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    surfaceScalarField phi(
        IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("phi", dimMass / dimTime, Zero));

    volScalarField progress(
        IOobject(
            "c_tilde",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    const volScalarField reactionRate(
        IOobject(
            "omega_c",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);

    const tmp<fvScalarMatrix> sourceMatrix =
        foamNordic::combustion::explicitSource(progress, reactionRate);
    const fvScalarMatrix& matrix = sourceMatrix();

    if (matrix.dimensions() != reactionRate.dimensions() * dimVolume) {
        throw std::runtime_error(
            "Progress-variable source matrix has incorrect dimensions.");
    }

    scalar localError = 0;
    scalar localScale = 1;
    forAll(matrix.source(), celli) {
        const scalar expected = mesh.V()[celli] * reactionRate[celli];
        localError = max(localError, mag(matrix.source()[celli] - expected));
        localScale = max(localScale, mag(expected));
    }
    reduce(localError, maxOp<scalar>());
    reduce(localScale, maxOp<scalar>());

    if (!std::isfinite(localError) || localError > 1e-12 * localScale) {
        throw std::runtime_error(
            "Positive explicit source was not assembled on the right-hand side.");
    }

    volScalarField specificRate(
        IOobject(
            "omega_specific",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("omega_specific", dimless / dimTime, 0.25));
    volScalarField density(
        IOobject(
            "rho_oracle",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("rho_oracle", dimDensity, 1.0));
    forAll(specificRate, celli) {
        specificRate[celli] = 0.25 + 1e-3 * scalar(celli % 17);
        density[celli] = 0.8 + 2e-3 * scalar(celli % 31);
    }
    const tmp<fvScalarMatrix> specificMatrix =
        foamNordic::combustion::explicitSpecificSource(
            progress, specificRate, density);
    if (specificMatrix().dimensions() != dimMass / dimTime) {
        throw std::runtime_error(
            "Specific progress-rate source matrix has incorrect dimensions.");
    }

    scalar localSpecificError = 0;
    scalar localSpecificScale = 1;
    forAll(specificMatrix().source(), celli) {
        const scalar expected =
            mesh.V()[celli] * density[celli] * specificRate[celli];
        localSpecificError = max(
            localSpecificError,
            mag(specificMatrix().source()[celli] - expected));
        localSpecificScale = max(localSpecificScale, mag(expected));
    }
    reduce(localSpecificError, maxOp<scalar>());
    reduce(localSpecificScale, maxOp<scalar>());
    if (!std::isfinite(localSpecificError)
        || localSpecificError > 1e-12 * localSpecificScale) {
        throw std::runtime_error(
            "Specific progress rate was not assembled as V*rho*omega.");
    }

    Info<< "[FoamNordic] Progress-variable source matrix: PASS" << nl
        << "[FoamNordic] Maximum volumetric source error: " << localError << nl
        << "[FoamNordic] Maximum specific source error: "
        << localSpecificError << nl;
    return 0;
}
