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
#include "frame.H"

#include "fvcCurl.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvcLaplacian.H"
#include "IOdictionary.H"
#include "dimensionedScalar.H"
#include "surfaceFields.H"

#include <span>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
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

std::optional<scalar> scalarConstant(const Expression& expression) {
    if (!expression.primitive()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtod(expression.name.c_str(), &end);
    if (errno != 0 || end == expression.name.c_str() || *end != '\0'
        || !std::isfinite(value)) {
        return std::nullopt;
    }
    return static_cast<scalar>(value);
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
    if (name == "deltaTime") {
        return tmp<volScalarField>::New(
            IOobject(
                "FoamNordic:deltaTime",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false),
            mesh_,
            mesh_.time().deltaT());
    }
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
            } else if (expression.name == "sqrt") {
                if constexpr (scalarField<FieldType>) {
                    return Foam::sqrt(value);
                }
            } else if (expression.name == "abs") {
                if constexpr (scalarField<FieldType>) {
                    return Foam::mag(value);
                }
            } else if (expression.name == "trace") {
                if constexpr (tensorField<FieldType>) {
                    return Foam::tr(value);
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

FieldValue OperationFrame::evaluateAlgebra(const Expression& expression) {
    const auto leftConstant = scalarConstant(expression.arguments[0]);
    const auto rightConstant = scalarConstant(expression.arguments[1]);
    if (expression.name == "greater" && leftConstant != rightConstant) {
        const auto& argument = evaluate(
            expression.arguments[leftConstant ? 1 : 0]);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                const auto& value = field(stored);
                using FieldType = std::decay_t<decltype(value)>;
                if constexpr (scalarField<FieldType>) {
                    const dimensionedScalar threshold(
                        "FoamNordic:greater",
                        value.dimensions(),
                        leftConstant ? *leftConstant : *rightConstant);
                    return leftConstant
                        ? Foam::pos(threshold - value)
                        : Foam::pos(value - threshold);
                }
                throw std::invalid_argument(
                    "OpenFOAM greater-than requires a scalar field.");
            },
            argument);
    }
    if (expression.name == "multiply" && leftConstant != rightConstant) {
        const auto coefficient = leftConstant ? *leftConstant : *rightConstant;
        const auto& argument = evaluate(
            expression.arguments[leftConstant ? 1 : 0]);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                return coefficient * field(stored);
            },
            argument);
    }
    if (expression.name == "divide" && rightConstant && !leftConstant) {
        const auto& argument = evaluate(expression.arguments[0]);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                return field(stored) / *rightConstant;
            },
            argument);
    }
    if (expression.name == "divide" && leftConstant && !rightConstant) {
        const auto& argument = evaluate(expression.arguments[1]);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                const auto& value = field(stored);
                using FieldType = std::decay_t<decltype(value)>;
                if constexpr (scalarField<FieldType>) {
                    const dimensionedScalar numerator(
                        "FoamNordic:numerator", dimless, *leftConstant);
                    return numerator / value;
                }
                throw std::invalid_argument(
                    "OpenFOAM constant division requires a scalar field.");
            },
            argument);
    }
    if ((expression.name == "maximum" || expression.name == "minimum")
        && leftConstant != rightConstant) {
        const auto coefficient = leftConstant ? *leftConstant : *rightConstant;
        const auto& argument = evaluate(
            expression.arguments[leftConstant ? 1 : 0]);
        return std::visit(
            [&](const auto& stored) -> FieldValue {
                const auto& value = field(stored);
                using FieldType = std::decay_t<decltype(value)>;
                if constexpr (scalarField<FieldType>) {
                    const dimensionedScalar bound(
                        "FoamNordic:bound",
                        value.dimensions(),
                        coefficient);
                    return expression.name == "maximum"
                        ? Foam::max(value, bound)
                        : Foam::min(value, bound);
                }
                throw std::invalid_argument(
                    "OpenFOAM minimum and maximum require a scalar field.");
            },
            argument);
    }
    if ((expression.name == "add" || expression.name == "subtract"
         || expression.name == "maximum" || expression.name == "minimum"
         || expression.name == "greater")
        && !leftConstant && !rightConstant) {
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
                        if constexpr (std::is_same_v<Left, Right>) {
                            if (expression.name == "add") {
                                return left + right;
                            }
                            if (expression.name == "subtract") {
                                return left - right;
                            }
                            if constexpr (scalarField<Left>) {
                                if (expression.name == "greater") {
                                    return Foam::pos(left - right);
                                }
                                return expression.name == "maximum"
                                    ? Foam::max(left, right)
                                    : Foam::min(left, right);
                            }
                        }
                        throw std::invalid_argument(
                            "OpenFOAM algebra requires matching field types.");
                    },
                    rightArgument);
            },
            leftArgument);
    }
    if ((expression.name == "multiply" || expression.name == "divide")
        && !leftConstant && !rightConstant) {
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
                        if constexpr (scalarField<Left>) {
                            if (expression.name == "multiply") {
                                return left * right;
                            }
                        }
                        if constexpr (scalarField<Right>) {
                            if (expression.name == "multiply") {
                                return left * right;
                            }
                            return left / right;
                        }
                        throw std::invalid_argument(
                            "OpenFOAM field multiplication and division "
                            "require at least one scalar field.");
                    },
                    rightArgument);
            },
            leftArgument);
    }
    throw std::invalid_argument(
        "OpenFOAM " + expression.name
        + " requires compatible fields or one finite scalar constant.");
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
                    } else if (expression.name == "outer") {
                        if constexpr (vectorField<Left> && vectorField<Right>) {
                            return left * right;
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
    if (expression.name == "div" && !expression.arguments[0].primitive()) {
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
        if (!expression.arguments[0].primitive()) {
            const auto& coefficientArgument = evaluate(expression.arguments[0]);
            return std::visit(
                [&](const auto& stored) -> FieldValue {
                    const auto& coefficient = field(stored);
                    using Coefficient = std::decay_t<decltype(coefficient)>;
                    if constexpr (scalarField<Coefficient>) {
                        return apply(coefficient);
                    }
                    throw std::invalid_argument(
                        "OpenFOAM laplacian coefficient expression must "
                        "resolve to a scalar field.");
                },
                coefficientArgument);
        }
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
    if (expression.name == "dot" || expression.name == "ddot"
        || expression.name == "outer") {
        return evaluateContraction(expression);
    }
    if (expression.name == "div" || expression.name == "laplacian") {
        return evaluateCoefficient(expression);
    }
    if (expression.name == "add" || expression.name == "subtract"
        || expression.name == "multiply" || expression.name == "divide"
        || expression.name == "maximum" || expression.name == "minimum"
        || expression.name == "greater") {
        return evaluateAlgebra(expression);
    }
    throw std::invalid_argument(
        "Unsupported binary OpenFOAM operation: " + expression.name);
}

FieldValue OperationFrame::evaluateTernary(const Expression& expression) {
    if (expression.name != "where" || expression.arguments.size() != 3) {
        throw std::invalid_argument(
            "Unsupported ternary OpenFOAM operation: " + expression.name);
    }
    const auto& conditionArgument = evaluate(expression.arguments[0]);
    const auto trueConstant = scalarConstant(expression.arguments[1]);
    const auto falseConstant = scalarConstant(expression.arguments[2]);
    if (trueConstant && falseConstant) {
        throw std::invalid_argument(
            "OpenFOAM where requires at least one field branch.");
    }
    return std::visit(
        [&](const auto& storedCondition) -> FieldValue {
            const auto& condition = field(storedCondition);
            using Condition = std::decay_t<decltype(condition)>;
            if constexpr (!scalarField<Condition>) {
                throw std::invalid_argument(
                    "OpenFOAM where condition must be a scalar field.");
            } else {
                const dimensionedScalar one("FoamNordic:one", dimless, 1.0);
                if (trueConstant || falseConstant) {
                    const auto fieldIndex = trueConstant ? 2 : 1;
                    const auto& fieldArgument = evaluate(
                        expression.arguments[fieldIndex]);
                    const auto constant = trueConstant
                        ? *trueConstant : *falseConstant;
                    return std::visit(
                        [&](const auto& stored) -> FieldValue {
                            const auto& value = field(stored);
                            using Value = std::decay_t<decltype(value)>;
                            if constexpr (scalarField<Value>) {
                                const dimensionedScalar branchConstant(
                                    "FoamNordic:where",
                                    value.dimensions(),
                                    constant);
                                if (trueConstant) {
                                    return condition * branchConstant
                                        + (one - condition) * value;
                                }
                                return condition * value
                                    + (one - condition) * branchConstant;
                            }
                            throw std::invalid_argument(
                                "OpenFOAM where scalar branches require a scalar field.");
                        },
                        fieldArgument);
                }
                const auto& trueArgument = evaluate(expression.arguments[1]);
                const auto& falseArgument = evaluate(expression.arguments[2]);
                return std::visit(
                    [&](const auto& storedTrue) -> FieldValue {
                        const auto& trueValue = field(storedTrue);
                        using True = std::decay_t<decltype(trueValue)>;
                        return std::visit(
                            [&](const auto& storedFalse) -> FieldValue {
                                const auto& falseValue = field(storedFalse);
                                using False = std::decay_t<decltype(falseValue)>;
                                if constexpr (std::is_same_v<True, False>) {
                                    return condition * trueValue
                                        + (one - condition) * falseValue;
                                }
                                throw std::invalid_argument(
                                    "OpenFOAM where branches require matching field types.");
                            },
                            falseArgument);
                    },
                    trueArgument);
            }
        },
        conditionArgument);
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
        if (expression.arguments.size() == 3) {
            return evaluateTernary(expression);
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

const volScalarField& OperationFrame::scalar(const Expression& expression) {
    const auto& value = evaluate(expression);
    return std::visit(
        [](const auto& stored) -> const volScalarField& {
            const auto& resolved = field(stored);
            using FieldType = std::decay_t<decltype(resolved)>;
            if constexpr (scalarField<FieldType>) {
                return resolved;
            }
            throw std::invalid_argument(
                "OpenFOAM expression must resolve to a scalar field.");
        },
        value);
}

const volScalarField& OperationFrame::scalar(const std::string& expression) {
    return scalar(Expression::parse(expression));
}

std::string OperationFrame::snapshot(
    const Expression& expression,
    const std::size_t index) {
    const auto& value = evaluate(expression);
    const auto name = "FoamNordic:outputSnapshot:" + std::to_string(index);
    std::visit(
        [&](const auto& stored) {
            const auto& source = field(stored);
            using FieldType = std::decay_t<decltype(source)>;
            if (!fields_.emplace(name, tmp<FieldType>::New(source)).second) {
                throw std::invalid_argument(
                    "OpenFOAM native algebra output snapshot is not unique.");
            }
        },
        value);
    return name;
}

void OperationFrame::validateAssignment(
    const word& name,
    const Expression& expression) {
    const auto& value = evaluate(expression);
    std::visit(
        [&](const auto& stored) {
            const auto& source = field(stored);
            using FieldType = std::decay_t<decltype(source)>;
            if (!mesh_.foundObject<FieldType>(name)) {
                throw std::invalid_argument(
                    "OpenFOAM native algebra output "
                    + std::string(name.c_str())
                    + " does not match its expression field type.");
            }
            const auto& target = mesh_.lookupObject<FieldType>(name);
            if (target.dimensions() != source.dimensions()) {
                throw std::invalid_argument(
                    "OpenFOAM native algebra output "
                    + std::string(name.c_str())
                    + " does not match its expression dimensions.");
            }
            for (const auto& item : source.primitiveField()) {
                if (!std::isfinite(static_cast<double>(Foam::mag(item)))) {
                    throw std::invalid_argument(
                        "OpenFOAM native algebra output "
                        + std::string(name.c_str())
                        + " contains a non-finite value.");
                }
            }
        },
        value);
}

void OperationFrame::assign(
    const word& name,
    const Expression& expression) {
    const auto& value = evaluate(expression);
    std::visit(
        [&](const auto& stored) {
            const auto& source = field(stored);
            using FieldType = std::decay_t<decltype(source)>;
            auto& target = mesh_.lookupObjectRef<FieldType>(name);
            target.primitiveFieldRef() = source.primitiveField();
        },
        value);
}

void OperationFrame::correctBoundary(
    const word& name,
    const Expression& expression) {
    const auto& value = evaluate(expression);
    std::visit(
        [&](const auto& stored) {
            using FieldType = std::decay_t<decltype(field(stored))>;
            mesh_.lookupObjectRef<FieldType>(name).correctBoundaryConditions();
        },
        value);
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
