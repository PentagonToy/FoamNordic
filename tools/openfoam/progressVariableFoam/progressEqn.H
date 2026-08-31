Foam::fvScalarMatrix progressEqn(
    Foam::fvm::ddt(rho, progress)
  + Foam::fvm::div(phi, progress)
  - Foam::fvm::laplacian(turbulence->muEff() / progressSchmidt, progress)
 ==
    combustion->R(progress)
  + fvOptions(rho, progress));

progressEqn.relax();
fvOptions.constrain(progressEqn);
progressEqn.solve();
fvOptions.correct(progress);
progress.clamp_range(Foam::scalar(0), Foam::scalar(1));
progress.correctBoundaryConditions();

Foam::Info<< "progress min/max = " << Foam::gMin(progress) << ", "
          << Foam::gMax(progress) << Foam::nl;
