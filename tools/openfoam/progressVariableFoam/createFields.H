Foam::Info<< "Reading thermophysical properties\n" << Foam::endl;
Foam::autoPtr<Foam::psiReactionThermo> pThermo(
    Foam::psiReactionThermo::New(mesh));
Foam::psiReactionThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

Foam::basicSpecieMixture& composition = thermo.composition();
const Foam::word inertSpecie(thermo.get<Foam::word>("inertSpecie"));
if (!composition.species().found(inertSpecie)) {
    FatalIOErrorIn(args.executable().c_str(), thermo)
        << "Inert specie " << inertSpecie
        << " not found in available species " << composition.species()
        << Foam::exit(Foam::FatalIOError);
}

Foam::volScalarField rho(
    Foam::IOobject(
        "rho",
        runTime.timeName(),
        mesh,
        Foam::IOobject::READ_IF_PRESENT,
        Foam::IOobject::AUTO_WRITE),
    thermo.rho());

Foam::volVectorField U(
    Foam::IOobject(
        "U",
        runTime.timeName(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::AUTO_WRITE),
    mesh);
Foam::volScalarField& p = thermo.p();

Foam::IOdictionary transportProperties(
    Foam::IOobject(
        "progressVariableTransportProperties",
        runTime.constant(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE));
const Foam::scalar progressSchmidt =
    transportProperties.getOrDefault<Foam::scalar>("progressSchmidt", 1.0);
const Foam::scalar varianceSchmidt =
    transportProperties.getOrDefault<Foam::scalar>("varianceSchmidt", 1.0);
const Foam::scalar varianceDissipationCoefficient =
    transportProperties.getOrDefault<Foam::scalar>(
        "varianceDissipationCoefficient", 2.0);
if (
    progressSchmidt <= 0
 || varianceSchmidt <= 0
 || varianceDissipationCoefficient < 0) {
    FatalIOErrorInFunction(transportProperties)
        << "progressSchmidt and varianceSchmidt must be positive and "
        << "varianceDissipationCoefficient must be non-negative"
        << Foam::exit(Foam::FatalIOError);
}

// The portable reference solver uses the same cube-root-volume filter width
// as its supported LES cases. Production solvers may replace this field with
// their turbulence model's native delta while retaining the equation contract.
Foam::volScalarField filterWidth(
    Foam::IOobject(
        "progressVariableDelta",
        runTime.timeName(),
        mesh,
        Foam::IOobject::NO_READ,
        Foam::IOobject::NO_WRITE),
    mesh,
    Foam::dimensionedScalar("delta", Foam::dimLength, Foam::Zero));
forAll(filterWidth, celli) {
    filterWidth[celli] = Foam::cbrt(mesh.V()[celli]);
}
forAll(filterWidth.boundaryField(), patchi) {
    filterWidth.boundaryFieldRef()[patchi] =
        filterWidth.boundaryField()[patchi].patchInternalField();
}

Foam::IOdictionary combustionDictionary(
    Foam::IOobject(
        "combustionProperties",
        runTime.constant(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        false));
const Foam::word combustionModel =
    combustionDictionary.get<Foam::word>("combustionModel");
const Foam::dictionary& combustionCoeffs =
    combustionDictionary.subDict(combustionModel + "Coeffs");
const Foam::word progressName =
    combustionCoeffs.get<Foam::word>("progressField");
const Foam::word sourceName =
    combustionCoeffs.get<Foam::word>("reactionRateField");
const Foam::dimensionSet sourceDimensions =
    combustionCoeffs.get<Foam::dimensionSet>("reactionRateDimensions");
const Foam::word varianceName =
    transportProperties.get<Foam::word>("varianceField");

Foam::volScalarField progress(
    Foam::IOobject(
        progressName,
        runTime.timeName(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::AUTO_WRITE),
    mesh);
Foam::volScalarField variance(
    Foam::IOobject(
        varianceName,
        runTime.timeName(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::AUTO_WRITE),
    mesh);
Foam::volScalarField reactionRate(
    Foam::IOobject(
        sourceName,
        runTime.timeName(),
        mesh,
        Foam::IOobject::READ_IF_PRESENT,
        Foam::IOobject::AUTO_WRITE),
    mesh,
    Foam::dimensionedScalar(sourceDimensions, Foam::Zero));

#include "compressibleCreatePhi.H"

Foam::pressureControl pressureControl(p, rho, pimple.dict(), false);
mesh.setFluxRequired(p.name());

Foam::Info<< "Creating turbulence model\n" << Foam::endl;
Foam::autoPtr<Foam::compressible::turbulenceModel> turbulence(
    Foam::compressible::turbulenceModel::New(rho, U, phi, thermo));

Foam::Info<< "Creating FoamNordic combustion model\n" << Foam::endl;
Foam::autoPtr<Foam::CombustionModel<Foam::psiReactionThermo>> combustion(
    Foam::CombustionModel<Foam::psiReactionThermo>::New(thermo, turbulence()));

// Prime the lagged source and table state before the first scalar solve.
combustion->correct();
rho = thermo.rho();

const Foam::volScalarField& psi = thermo.psi();
const Foam::volScalarField& T = thermo.T();
static_cast<void>(T);

#include "createDpdt.H"
#include "createK.H"
#include "createMRF.H"

const Foam::dimensionedScalar rhoMax(
    "rhoMax", Foam::dimDensity, Foam::GREAT, pimple.dict());
const Foam::dimensionedScalar rhoMin(
    "rhoMin", Foam::dimDensity, Foam::Zero, pimple.dict());

#include "createFvOptions.H"
