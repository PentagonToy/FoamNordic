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
#include "fvcLaplacian.H"

#include "closureHook.H"
#include "fieldBridge.H"
#include "operations/frame.H"

#include <cstring>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<class Value>
void scaleReference(std::vector<std::byte>& bytes, double scale) {
    for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(Value)) {
        Value value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(Value));
        value = static_cast<Value>(static_cast<double>(value) * scale);
        std::memcpy(bytes.data() + offset, &value, sizeof(Value));
    }
}

template<class Value>
void seedField(std::span<std::byte> bytes, double seed) {
    for (std::size_t offset = 0, component = 0;
         offset < bytes.size();
         offset += sizeof(Value), ++component) {
        const auto value = static_cast<Value>(
            seed * static_cast<double>(1 + component % 97));
        std::memcpy(bytes.data() + offset, &value, sizeof(Value));
    }
}

void seedField(
    foamnordic::fjord::MutableTensorView field,
    double seed) {
    field.validate();
    switch (field.element) {
        case foamnordic::fjord::Element::float32:
            seedField<float>(field.bytes, seed);
            return;
        case foamnordic::fjord::Element::float64:
            seedField<double>(field.bytes, seed);
            return;
        case foamnordic::fjord::Element::int32:
        case foamnordic::fjord::Element::int64:
            throw std::invalid_argument(
                "FoamNordic closure probe seeding requires floating-point "
                "storage.");
    }
}

void scaleReference(
    std::vector<std::byte>& bytes,
    foamnordic::fjord::Element element,
    double scale) {
    switch (element) {
        case foamnordic::fjord::Element::float32:
            scaleReference<float>(bytes, scale);
            return;
        case foamnordic::fjord::Element::float64:
            scaleReference<double>(bytes, scale);
            return;
        case foamnordic::fjord::Element::int32:
        case foamnordic::fjord::Element::int64:
            throw std::invalid_argument(
                "FoamNordic closure probe scaling requires floating-point "
                "storage.");
    }
}

template<class Value>
bool hasNonzero(const std::vector<std::byte>& bytes) {
    for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(Value)) {
        Value value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(Value));
        if (value != Value{}) {
            return true;
        }
    }
    return false;
}

bool hasNonzero(
    const std::vector<std::byte>& bytes,
    foamnordic::fjord::Element element) {
    switch (element) {
        case foamnordic::fjord::Element::float32:
            return hasNonzero<float>(bytes);
        case foamnordic::fjord::Element::float64:
            return hasNonzero<double>(bytes);
        case foamnordic::fjord::Element::int32:
        case foamnordic::fjord::Element::int64:
            return false;
    }
    return false;
}

void requireIdentity(
    const foamnordic::fjord::TensorView& actual,
    const foamnordic::fjord::TensorView& expected) {
    actual.validate();
    expected.validate();
    if (actual.element != expected.element || actual.shape != expected.shape
        || actual.bytes.size() != expected.bytes.size()) {
        throw std::runtime_error(
            "FoamNordic closure output does not match the probe expression.");
    }
    if (std::memcmp(
            actual.bytes.data(), expected.bytes.data(), actual.bytes.size())
        != 0) {
        throw std::runtime_error(
            "FoamNordic closure output differs from the identity reference.");
    }
}

}  // namespace

using namespace Foam;

int main(int argc, char* argv[]) {
#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    Foam::volScalarField pressure(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE),
        mesh);
    Foam::volVectorField velocity(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE),
        mesh);
    Foam::IOdictionary configuration(
        Foam::IOobject(
            "foamnordicClosureDict",
            runTime.system(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            false));

    const auto probeExpression = configuration.get<string>("probeExpression");
    const auto probeOutput = configuration.get<word>("probeOutput");
    auto probePatch = configuration.getOrDefault<word>(
        "probePatch", word());
    if (probePatch == "none") {
        probePatch.clear();
    }
    const auto probeScale = configuration.getOrDefault<scalar>(
        "probeScale", 1.0);
    const auto probeSeed = configuration.getOrDefault<scalar>(
        "probeSeed", 0.0);
    const auto probeExpectFailure = configuration.getOrDefault<bool>(
        "probeExpectFailure", false);
    if (!std::isfinite(probeScale) || !std::isfinite(probeSeed)) {
        throw std::invalid_argument(
            "FoamNordic probeScale and probeSeed must be finite.");
    }
    if (probeSeed != 0.0) {
        seedField(probePatch.empty()
            ? Foam::foamNordic::outputFieldView(
                mesh, probeOutput, 0, runTime.value())
            : Foam::foamNordic::outputPatchView(
                mesh, probeOutput, probePatch, 0, runTime.value()),
            probeSeed);
        if (probePatch.empty()) {
            Foam::foamNordic::correctFieldBoundary(mesh, probeOutput);
        }
        Foam::Info
            << "[FoamNordic] Seeded " << probeOutput
            << " in memory for deterministic verification" << Foam::nl;
    }
    Foam::foamNordic::ClosureHook closure(configuration);
    if (probeExpectFailure) {
        const auto beforeView = probePatch.empty()
            ? Foam::foamNordic::inputFieldView(
                mesh, probeOutput, 0, runTime.value())
            : Foam::foamNordic::inputPatchView(
                mesh, probeOutput, probePatch, 0, runTime.value());
        std::vector<std::byte> before(
            beforeView.bytes.begin(), beforeView.bytes.end());
        auto ownedBefore = beforeView;
        ownedBefore.bytes = std::span<const std::byte>(before);
        try {
            static_cast<void>(closure.invoke(mesh, runTime));
        } catch (const std::exception&) {
            const auto after = probePatch.empty()
                ? Foam::foamNordic::inputFieldView(
                    mesh, probeOutput, 0, runTime.value())
                : Foam::foamNordic::inputPatchView(
                    mesh, probeOutput, probePatch, 0, runTime.value());
            requireIdentity(after, ownedBefore);
            Foam::Info
                << "[FoamNordic] Rejected closure left " << probeOutput
                << " unchanged: PASS" << Foam::nl;
            return 0;
        }
        throw std::runtime_error(
            "FoamNordic closure probe expected the worker to reject its "
            "exchange.");
    }
    for (std::uint64_t expectedIndex = 0; expectedIndex < 2; ++expectedIndex) {
        Foam::foamNordic::operations::OperationFrame reference(mesh);
        const auto expectedView = probePatch.empty()
            ? reference.view("reference", probeExpression.c_str())
            : Foam::foamNordic::inputPatchView(
                mesh,
                word(probeExpression),
                probePatch,
                expectedIndex,
                runTime.value());
        std::vector<std::byte> expected(
            expectedView.bytes.begin(), expectedView.bytes.end());
        if (probeScale != 1.0
            && !hasNonzero(expected, expectedView.element)) {
            throw std::runtime_error(
                "FoamNordic non-identity closure probe requires a non-zero "
                "reference field.");
        }
        scaleReference(expected, expectedView.element, probeScale);
        auto ownedExpected = expectedView;
        ownedExpected.bytes = std::span<const std::byte>(expected);

        const auto exchangeIndex = closure.invoke(mesh, runTime);
        const auto actual = probePatch.empty()
            ? Foam::foamNordic::inputFieldView(
                mesh, probeOutput, exchangeIndex, runTime.value())
            : Foam::foamNordic::inputPatchView(
                mesh,
                probeOutput,
                probePatch,
                exchangeIndex,
                runTime.value());
        requireIdentity(actual, ownedExpected);
        if (exchangeIndex != expectedIndex) {
            throw std::runtime_error(
                "FoamNordic closure hook issued an incorrect exchange index.");
        }
        Foam::Info
            << "[FoamNordic] Closure invocation " << exchangeIndex
            << ": exact expected result" << Foam::nl;
    }

    closure.shutdown();
    Foam::Info << "[FoamNordic] OpenFOAM closure hook: PASS" << Foam::nl;
    return 0;
}
