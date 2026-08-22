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

    Info<< "[FoamNordic] Progress-variable source matrix: PASS" << nl
        << "[FoamNordic] Maximum source error: " << localError << nl;
    return 0;
}
