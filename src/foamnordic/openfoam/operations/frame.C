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

#include "frame.H"

#include "fvcCurl.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvcLaplacian.H"
#include "IOdictionary.H"
#include "dimensionedScalar.H"
#include "surfaceFields.H"

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace Foam::foamNordic::operations {
namespace {

template<class Type>
inline constexpr bool scalarField = std::is_same_v<Type, volScalarField>;
template<class Type>
inline constexpr bool vectorField = std::is_same_v<Type, volVectorField>;
template<class Type>
inline constexpr bool tensorField =
    std::is_same_v<Type, volTensorField>
    || std::is_same_v<Type, volSymmTensorField>
    || std::is_same_v<Type, volSphericalTensorField>;
template<class Type>
inline constexpr bool contractionTensorField =
    std::is_same_v<Type, volTensorField>
    || std::is_same_v<Type, volSymmTensorField>;

template<class Type>
struct ReferenceWrapper : std::false_type {};
template<class Type>
struct ReferenceWrapper<std::reference_wrapper<Type>> : std::true_type {};

template<class Stored>
decltype(auto) field(const Stored& stored) {
    if constexpr (ReferenceWrapper<std::decay_t<Stored>>::value) {
        return stored.get();
    } else {
        return stored();
    }
}

foamnordic::fjord::Element scalarElement() {
    if constexpr (sizeof(scalar) == sizeof(double)) {
        return foamnordic::fjord::Element::float64;
    }
    return foamnordic::fjord::Element::float32;
}

template<class Value>
foamnordic::fjord::TensorView tensorView(
    const std::string& key,
    const Field<Value>& values) {
    constexpr auto components =
        static_cast<std::uint64_t>(pTraits<Value>::nComponents);
    static_assert(sizeof(Value) == components * sizeof(scalar));
    std::vector<std::uint64_t> shape{
        static_cast<std::uint64_t>(values.size())};
    if constexpr (components > 1) {
        shape.push_back(components);
    }
    return {
        key,
        scalarElement(),
        std::move(shape),
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(values.cdata()),
            static_cast<std::size_t>(values.size()) * sizeof(Value)),
        0,
        0.0,
    };
}

}  // namespace

OperationFrame::OperationFrame(const fvMesh& mesh) : mesh_(mesh) {}

namespace {

template<class FieldType>
void bindField(
    std::unordered_map<std::string, FieldValue>& fields,
    const std::string& name,
    const FieldType& value) {
    if (name.empty() || !fields.emplace(name, std::cref(value)).second) {
        throw std::invalid_argument(
            "FoamNordic operation bindings must be non-empty and unique.");
    }
}

}  // namespace

void OperationFrame::bind(
    const std::string& name,
    const volScalarField& value) {
    bindField(fields_, name, value);
}

void OperationFrame::bind(
    const std::string& name,
    const volVectorField& value) {
    bindField(fields_, name, value);
}

void OperationFrame::bind(
    const std::string& name,
    const volTensorField& value) {
    bindField(fields_, name, value);
}

void OperationFrame::bind(
    const std::string& name,
    const volSymmTensorField& value) {
    bindField(fields_, name, value);
}

void OperationFrame::bind(
    const std::string& name,
    const volSphericalTensorField& value) {
    bindField(fields_, name, value);
}

FieldValue OperationFrame::lookup(const std::string& name) const {
    const word fieldName(name);
#define FOAMNORDIC_LOOKUP_FIELD(FIELD_TYPE)                    \
    if (mesh_.foundObject<FIELD_TYPE>(fieldName)) {            \
        return std::cref(mesh_.lookupObject<FIELD_TYPE>(fieldName)); \
    }
    FOAMNORDIC_LOOKUP_FIELD(volScalarField)
    FOAMNORDIC_LOOKUP_FIELD(volVectorField)
    FOAMNORDIC_LOOKUP_FIELD(volTensorField)
    FOAMNORDIC_LOOKUP_FIELD(volSymmTensorField)
    FOAMNORDIC_LOOKUP_FIELD(volSphericalTensorField)
#undef FOAMNORDIC_LOOKUP_FIELD
    throw std::invalid_argument(
        "OpenFOAM objectRegistry has no supported volume field named "
        + name + '.');
}

FieldValue OperationFrame::evaluateUnary(const Expression& expression) {
    if (expression.arguments.size() != 1) {
        throw std::invalid_argument(
            "OpenFOAM operation " + expression.name
            + " requires exactly one argument.");
    }
    const auto& argument = evaluate(expression.arguments.front());
    return std::visit(
        [&](const auto& stored) -> FieldValue {
            const auto& value = field(stored);
            using FieldType = std::decay_t<decltype(value)>;
            if (expression.name == "grad") {
                if constexpr (scalarField<FieldType> || vectorField<FieldType>) {
                    return fvc::grad(value);
                }
            } else if (expression.name == "div") {
                if constexpr (vectorField<FieldType> || tensorField<FieldType>) {
                    return fvc::div(value);
                }
            } else if (expression.name == "curl") {
                if constexpr (vectorField<FieldType>) {
                    return fvc::curl(value);
                }
            } else if (expression.name == "laplacian") {
                if constexpr (
                    scalarField<FieldType> || vectorField<FieldType>
                    || tensorField<FieldType>) {
                    return fvc::laplacian(value);
                }
            } else if (expression.name == "mag") {
                if constexpr (
                    scalarField<FieldType> || vectorField<FieldType>
                    || tensorField<FieldType>) {
                    return Foam::mag(value);
                }
            } else if (expression.name == "symm") {
                if constexpr (std::is_same_v<FieldType, volTensorField>) {
                    return Foam::symm(value);
                }
            } else if (expression.name == "dev") {
                if constexpr (
                    std::is_same_v<FieldType, volTensorField>
                    || std::is_same_v<FieldType, volSymmTensorField>) {
                    return Foam::dev(value);
                }
            } else {
                throw std::invalid_argument(
                    "Unsupported OpenFOAM operation: " + expression.name);
            }
            throw std::invalid_argument(
                "OpenFOAM operation " + expression.name
                + " does not accept the resolved field type.");
        },
        argument);
}

FieldValue OperationFrame::evaluateContraction(
    const Expression& expression) {
    const auto& leftArgument = evaluate(expression.arguments[0]);
    const auto& rightArgument = evaluate(expression.arguments[1]);
    return std::visit(
        [&](const auto& storedLeft) -> FieldValue {
            const auto& left = field(storedLeft);
            using Left = std::decay_t<decltype(left)>;
            return std::visit(
                [&](const auto& storedRight) -> FieldValue {
                    const auto& right = field(storedRight);
                    using Right = std::decay_t<decltype(right)>;
                    if (expression.name == "dot") {
                        if constexpr (
                            (vectorField<Left> && vectorField<Right>)
                            || (contractionTensorField<Left>
                                && vectorField<Right>)
                            || (vectorField<Left>
                                && contractionTensorField<Right>)
                            || (contractionTensorField<Left>
                                && contractionTensorField<Right>)) {
                            return left & right;
                        }
                    } else if (expression.name == "ddot") {
                        if constexpr (
                            contractionTensorField<Left>
                            && contractionTensorField<Right>) {
                            return left && right;
                        }
                    }
                    throw std::invalid_argument(
                        "OpenFOAM contraction " + expression.name
                        + " does not accept the resolved field types.");
                },
                rightArgument);
        },
        leftArgument);
}

FieldValue OperationFrame::evaluateCoefficient(
    const Expression& expression) {
    if (!expression.arguments[0].primitive()) {
        throw std::invalid_argument(
            "The first argument of OpenFOAM " + expression.name
            + " must name a coefficient field or property.");
    }
    const word coefficientName(expression.arguments[0].name);
    const auto& fieldArgument = evaluate(expression.arguments[1]);

    if (expression.name == "div") {
        if (!mesh_.foundObject<surfaceScalarField>(coefficientName)) {
            throw std::invalid_argument(
                "OpenFOAM objectRegistry has no surfaceScalarField named "
                + expression.arguments[0].name + '.');
        }
        const auto& flux =
            mesh_.lookupObject<surfaceScalarField>(coefficientName);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                const auto& value = field(stored);
                using FieldType = std::decay_t<decltype(value)>;
                if constexpr (
                    scalarField<FieldType> || vectorField<FieldType>
                    || tensorField<FieldType>) {
                    return fvc::div(flux, value);
                }
                throw std::invalid_argument(
                    "Binary OpenFOAM div requires a supported volume field.");
            },
            fieldArgument);
    }

    if (expression.name == "laplacian") {
        const auto apply = [&](const auto& coefficient) -> FieldValue {
            return std::visit(
                [&](const auto& stored) -> FieldValue {
                    const auto& value = field(stored);
                    using FieldType = std::decay_t<decltype(value)>;
                    if constexpr (
                        scalarField<FieldType> || vectorField<FieldType>
                        || tensorField<FieldType>) {
                        return fvc::laplacian(coefficient, value);
                    }
                    throw std::invalid_argument(
                        "Binary OpenFOAM laplacian requires a supported "
                        "volume field.");
                },
                fieldArgument);
        };
        if (mesh_.foundObject<volScalarField>(coefficientName)) {
            return apply(mesh_.lookupObject<volScalarField>(coefficientName));
        }
        for (const word& dictionaryName :
             {word("transportProperties"), word("physicalProperties")}) {
            IOobject object(
                dictionaryName,
                mesh_.time().constant(),
                mesh_,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE,
                false);
            if (!object.typeHeaderOk<IOdictionary>(false)) {
                continue;
            }
            const IOdictionary properties(object);
            if (properties.found(coefficientName)) {
                return apply(dimensionedScalar(coefficientName, properties));
            }
        }
        throw std::invalid_argument(
            "Could not resolve OpenFOAM laplacian coefficient "
            + expression.arguments[0].name + '.');
    }

    throw std::invalid_argument(
        "Unsupported binary OpenFOAM operation: " + expression.name);
}

FieldValue OperationFrame::evaluateBinary(const Expression& expression) {
    if (expression.arguments.size() != 2) {
        throw std::invalid_argument(
            "Binary OpenFOAM operation " + expression.name
            + " requires exactly two arguments.");
    }
    if (expression.name == "dot" || expression.name == "ddot") {
        return evaluateContraction(expression);
    }
    if (expression.name == "div" || expression.name == "laplacian") {
        return evaluateCoefficient(expression);
    }
    throw std::invalid_argument(
        "Unsupported binary OpenFOAM operation: " + expression.name);
}

const FieldValue& OperationFrame::evaluate(const Expression& expression) {
    const auto key = expression.canonical();
    if (const auto found = fields_.find(key); found != fields_.end()) {
        return found->second;
    }
    auto value = [&]() -> FieldValue {
        if (expression.primitive()) {
            return lookup(expression.name);
        }
        if (expression.arguments.size() == 1) {
            return evaluateUnary(expression);
        }
        if (expression.arguments.size() == 2) {
            return evaluateBinary(expression);
        }
        throw std::invalid_argument(
            "OpenFOAM operation " + expression.name
            + " has unsupported arity "
            + std::to_string(expression.arguments.size()) + '.');
    }();
    return fields_.emplace(key, std::move(value)).first->second;
}

const FieldValue& OperationFrame::evaluate(const std::string& expression) {
    return evaluate(Expression::parse(expression));
}

foamnordic::fjord::TensorView OperationFrame::view(
    const std::string& key,
    const std::string& expression) {
    const auto& value = evaluate(expression);
    return std::visit(
        [&](const auto& stored) {
            return tensorView(key, field(stored).primitiveField());
        },
        value);
}

void OperationFrame::provide(
    foamnordic::adapter::FieldInvocation& invocation,
    const std::string& key,
    const std::string& expression) {
    invocation.provide(view(key, expression));
}

}  // namespace Foam::foamNordic::operations
