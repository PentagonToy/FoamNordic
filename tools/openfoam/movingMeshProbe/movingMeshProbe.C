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

#include "dynamicFvMesh.H"
#include "fvCFD.H"

#include "closureHook.H"

#include <cstdint>
#include <stdexcept>

using namespace Foam;

int main(int argc, char* argv[]) {
#include "setRootCase.H"
#include "createTime.H"

    auto meshPointer = dynamicFvMesh::New(args, runTime);
    dynamicFvMesh& mesh = meshPointer();
    volVectorField probeField(
        IOobject(
            "probeField",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedVector("probe", dimless, vector(1, 2, 3)));
    IOdictionary configuration(
        IOobject(
            "foamnordicClosureDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    foamNordic::ClosureHook closure(configuration);
    const auto patch = configuration.get<word>("probePatch");
    const auto patchIndex = mesh.boundaryMesh().findPatchID(patch);
    if (patchIndex < 0) {
        throw std::invalid_argument(
            "FoamNordic moving-mesh probe patch does not exist.");
    }

    std::uint64_t invocations = 0;
    while (runTime.loop() && invocations < 3) {
        mesh.update();
        const auto faces = mesh.boundary()[patchIndex].size();
        const auto exchangeIndex = closure.invoke(mesh, runTime);
        if (exchangeIndex != invocations) {
            throw std::runtime_error(
                "FoamNordic moving-mesh exchange index is not monotonic.");
        }
        Info << "[FoamNordic] Moving mesh invocation " << exchangeIndex
             << ": cells=" << mesh.nCells()
             << ", patchFaces=" << faces << nl;
        ++invocations;
    }
    closure.shutdown();
    if (invocations != 3) {
        throw std::runtime_error(
            "FoamNordic moving-mesh probe completed too few updates.");
    }
    Info << "[FoamNordic] Moving mesh patch rebind: PASS" << nl;
    return 0;
}
