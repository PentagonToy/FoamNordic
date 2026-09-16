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

#include "closureHook.H"
#include "fvCFD.H"

int
main(
    int argc,
    char* argv[]
)
{
#include "setRootCaseLists.H"
#include "createTime.H"
#include "createMesh.H"

    Foam::volVectorField U(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );
    Foam::volScalarField p(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );
    Foam::IOdictionary properties(
        Foam::IOobject(
            "closureUpdateProperties",
            runTime.system(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        )
    );

    Foam::foamNordic::ClosureHook closure(properties.subDict("closure"));
    Foam::foamNordic::ClosureHook update(properties.subDict("update"));
    static_cast<void>(closure.invoke(mesh, runTime));
    static_cast<void>(update.invoke(mesh, runTime));
    closure.shutdown();
    update.shutdown();

    U.write();
    p.write();
    Foam::Info << "End\n" << Foam::endl;
    return 0;
}
