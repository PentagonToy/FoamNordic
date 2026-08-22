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

#include "operations/frame.H"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Statistics {
    double minimum{std::numeric_limits<double>::infinity()};
    double maximum{-std::numeric_limits<double>::infinity()};
};

template<class Value>
Statistics statistics(const foamnordic::fjord::TensorView& view) {
    Statistics result;
    for (std::size_t offset = 0; offset < view.bytes.size();
         offset += sizeof(Value)) {
        Value raw{};
        std::memcpy(&raw, view.bytes.data() + offset, sizeof(Value));
        const auto value = static_cast<double>(raw);
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "OpenFOAM operation produced a non-finite value: "
                + view.name);
        }
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
    }
    return result;
}

Statistics statistics(const foamnordic::fjord::TensorView& view) {
    view.validate();
    switch (view.element) {
        case foamnordic::fjord::Element::float32:
            return statistics<float>(view);
        case foamnordic::fjord::Element::float64:
            return statistics<double>(view);
        default:
            throw std::runtime_error(
                "OpenFOAM operation probe expected floating-point storage.");
    }
}

std::string shape(const std::vector<std::uint64_t>& extents) {
    std::string result("[");
    for (std::size_t index = 0; index < extents.size(); ++index) {
        if (index != 0) {
            result += ',';
        }
        result += std::to_string(extents[index]);
    }
    return result + ']';
}

}  // namespace

using namespace Foam;

int main(int argc, char* argv[]) {
    Foam::argList::noParallel();

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    Foam::volVectorField velocity(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE),
        mesh);
    Foam::volScalarField pressure(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE),
        mesh);

    Foam::foamNordic::operations::OperationFrame frame(mesh);
    const std::vector<std::pair<std::string, std::string>> expressions{
        {"grad_U", "grad(U)"},
        {"curl_U", "curl(U)"},
        {"laplacian_p", "laplacian(p)"},
        {"mag_grad_p", "mag(grad(p))"},
        {"deviatoric_strain", "dev(symm(grad(U)))"},
        {"strain_contraction", "ddot(grad(U),dev(symm(grad(U))))"},
    };

    Foam::Info
        << "[FoamNordic] OpenFOAM operation probe" << Foam::nl
        << "Expression                  Shape            Minimum           Maximum"
        << Foam::nl;

    for (const auto& [key, expression] : expressions) {
        const auto view = frame.view(key, expression);
        if (view.shape.front()
            != static_cast<std::uint64_t>(mesh.nCells())) {
            throw std::runtime_error(
                "OpenFOAM operation cell count does not match the mesh: "
                + expression);
        }
        const auto range = statistics(view);
        Foam::Info
            << expression << "    " << shape(view.shape) << "    "
            << range.minimum << "    " << range.maximum << Foam::nl;
    }

    Foam::Info << "[FoamNordic] OpenFOAM operations: PASS" << Foam::nl;
    return 0;
}
