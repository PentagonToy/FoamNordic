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

#ifndef Foam_foamNordic_operations_frame_H
#define Foam_foamNordic_operations_frame_H

#include "expression.H"

#include "fvMesh.H"
#include "volFields.H"

#include "foamnordic/backend/adapter/port.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>

namespace Foam::foamNordic::operations {

using FieldValue = std::variant<
    std::reference_wrapper<const volScalarField>,
    std::reference_wrapper<const volVectorField>,
    std::reference_wrapper<const volTensorField>,
    std::reference_wrapper<const volSymmTensorField>,
    std::reference_wrapper<const volSphericalTensorField>,
    tmp<volScalarField>,
    tmp<volVectorField>,
    tmp<volTensorField>,
    tmp<volSymmTensorField>,
    tmp<volSphericalTensorField>>;

class OperationFrame final {
    const fvMesh& mesh_;
    std::unordered_map<std::string, FieldValue> fields_;

    [[nodiscard]] FieldValue lookup(const std::string& name) const;
    [[nodiscard]] FieldValue evaluateUnary(const Expression& expression);
    [[nodiscard]] FieldValue evaluateBinary(const Expression& expression);
    [[nodiscard]] FieldValue evaluateContraction(
        const Expression& expression);
    [[nodiscard]] FieldValue evaluateCoefficient(
        const Expression& expression);

public:
    explicit OperationFrame(const fvMesh& mesh);

    OperationFrame(const OperationFrame&) = delete;
    OperationFrame& operator=(const OperationFrame&) = delete;

    [[nodiscard]] const FieldValue& evaluate(const Expression& expression);
    [[nodiscard]] const FieldValue& evaluate(const std::string& expression);

    void bind(const std::string& name, const volScalarField& value);
    void bind(const std::string& name, const volVectorField& value);
    void bind(const std::string& name, const volTensorField& value);
    void bind(const std::string& name, const volSymmTensorField& value);
    void bind(const std::string& name, const volSphericalTensorField& value);

    [[nodiscard]] foamnordic::fjord::TensorView view(
        const std::string& key,
        const std::string& expression);

    void provide(
        foamnordic::adapter::ClosureInvocation& invocation,
        const std::string& key,
        const std::string& expression);
};

}  // namespace Foam::foamNordic::operations

#endif
