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

#include "expression.H"

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid(const std::string& source) {
    try {
        static_cast<void>(
            Foam::foamNordic::operations::Expression::parse(source));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(
        "Invalid OpenFOAM expression was accepted: " + source);
}

}  // namespace

int main() {
    using Foam::foamNordic::operations::Expression;

    const auto primitive = Expression::parse(" U ");
    require(primitive.primitive(), "Primitive field was parsed as an operation.");
    require(primitive.canonical() == "U", "Primitive canonical name is wrong.");

    const auto gradient = Expression::parse(" grad( U ) ");
    require(!gradient.primitive(), "Gradient was parsed as a primitive field.");
    require(
        gradient.canonical() == "grad(U)",
        "Gradient canonical expression is wrong.");

    const auto nested = Expression::parse(
        "ddot(grad(U), dev(symm(grad(U))))");
    require(
        nested.canonical() == "ddot(grad(U),dev(symm(grad(U))))",
        "Nested canonical expression is wrong.");

    const auto coefficient = Expression::parse("laplacian(nu, U)");
    require(
        coefficient.arguments.size() == 2
            && coefficient.canonical() == "laplacian(nu,U)",
        "Binary OpenFOAM expression is wrong.");

    require_invalid("");
    require_invalid("grad()");
    require_invalid("grad(U");
    require_invalid("grad(U),p");
    require_invalid("laplacian(nu,,U)");
}
