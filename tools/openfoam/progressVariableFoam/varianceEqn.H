Foam::fvScalarMatrix varianceEqn(
    Foam::fvm::ddt(rho, variance)
  + Foam::fvm::div(phi, variance)
  - Foam::fvm::laplacian(turbulence->muEff() / varianceSchmidt, variance)
 ==
    fvOptions(rho, variance));

varianceEqn.relax();
fvOptions.constrain(varianceEqn);
varianceEqn.solve();
fvOptions.correct(variance);

variance.clamp_min(Foam::scalar(0));
forAll(variance, celli) {
    variance[celli] = Foam::min(
        variance[celli],
        progress[celli] * (Foam::scalar(1) - progress[celli]));
}
variance.correctBoundaryConditions();

Foam::Info<< "variance min/max = " << Foam::gMin(variance) << ", "
          << Foam::gMax(variance) << Foam::nl;
