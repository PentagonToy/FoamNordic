/*---------------------------------------------------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider: ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "CombustionModel.H"
#include "bound.H"
#include "fvCFD.H"
#include "fvOptions.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "psiReactionThermo.H"
#include "turbulentFluidThermoModel.H"

int main(int argc, char* argv[]) {
    Foam::argList::addNote(
        "Reference transient solver for FoamNordic progress-variable "
        "combustion coupling");

#include "setRootCaseLists.H"
#include "createTime.H"
#include "createMesh.H"
#include "createControl.H"
#include "createTimeControls.H"
#include "initContinuityErrs.H"
#include "createFields.H"

    turbulence->validate();

#include "compressibleCourantNo.H"
#include "setInitialDeltaT.H"

    Foam::Info<< "\nStarting FoamNordic progress-variable time loop\n"
              << Foam::endl;

    while (runTime.run()) {
#include "readTimeControls.H"

#include "compressibleCourantNo.H"
#include "setDeltaT.H"

        ++runTime;
        Foam::Info<< "Time = " << runTime.timeName() << Foam::nl << Foam::endl;

#include "rhoEqn.H"

        while (pimple.loop()) {
#include "UEqn.H"
#include "progressEqn.H"
#include "varianceEqn.H"

            // The updated moments produce the next lagged source, then the
            // manifold transaction, then one thermo.correct().
            combustion->correct();
            rho = thermo.rho();

            while (pimple.correct()) {
                if (pimple.consistent()) {
#include "pcEqn.H"
                } else {
#include "pEqn.H"
                }
            }

            if (pimple.turbCorr()) {
                turbulence->correct();
            }
        }

        rho = thermo.rho();
        runTime.write();
        runTime.printExecutionTime(Foam::Info);
    }

    Foam::Info<< "End\n" << Foam::endl;
    return 0;
}
