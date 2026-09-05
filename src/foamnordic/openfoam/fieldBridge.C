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

#include "fieldBridge.H"

#include "volFields.H"
#include "surfaceFields.H"

#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Foam::foamNordic {
namespace {

foamnordic::fjord::Element scalarElement() {
    if constexpr (sizeof(scalar) == sizeof(double)) {
        return foamnordic::fjord::Element::float64;
    }
    return foamnordic::fjord::Element::float32;
}

template<class Value>
std::vector<std::uint64_t> fieldShape(std::size_t cells) {
    constexpr auto components =
        static_cast<std::uint64_t>(pTraits<Value>::nComponents);
    static_assert(sizeof(Value) == components * sizeof(scalar));
    std::vector<std::uint64_t> shape{static_cast<std::uint64_t>(cells)};
    if constexpr (components > 1) {
        shape.push_back(components);
    }
    return shape;
}

template<class Value>
foamnordic::fjord::TensorView makeInputView(
    const word& name,
    const Field<Value>& field,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    return {
        name.c_str(),
        scalarElement(),
        fieldShape<Value>(field.size()),
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(field.cdata()),
            static_cast<std::size_t>(field.size()) * sizeof(Value)),
        exchangeIndex,
        physicalTime,
    };
}

template<class Value>
foamnordic::fjord::MutableTensorView makeOutputView(
    const word& name,
    Field<Value>& field,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    return {
        name.c_str(),
        scalarElement(),
        fieldShape<Value>(field.size()),
        std::span<std::byte>(
            reinterpret_cast<std::byte*>(field.data()),
            static_cast<std::size_t>(field.size()) * sizeof(Value)),
        exchangeIndex,
        physicalTime,
    };
}

template<class FieldType>
bool contains(const fvMesh& mesh, const word& name) {
    return mesh.foundObject<FieldType>(name);
}

template<class FieldType>
foamnordic::fjord::TensorView input(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    return makeInputView<typename FieldType::value_type>(
        name,
        mesh.lookupObject<FieldType>(name).primitiveField(),
        exchangeIndex,
        physicalTime);
}

template<class FieldType>
foamnordic::fjord::MutableTensorView output(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    return makeOutputView<typename FieldType::value_type>(
        name,
        mesh.lookupObjectRef<FieldType>(name).primitiveFieldRef(),
        exchangeIndex,
        physicalTime);
}

template<class FieldType>
void correct(const fvMesh& mesh, const word& name) {
    mesh.lookupObjectRef<FieldType>(name).correctBoundaryConditions();
}

label patchIndex(const fvMesh& mesh, const word& patch) {
    const auto index = mesh.boundaryMesh().findPatchID(patch);
    if (index < 0) {
        throw std::invalid_argument(
            "OpenFOAM mesh has no boundary patch named "
            + std::string(patch.c_str()) + '.');
    }
    return index;
}

template<class FieldType>
foamnordic::fjord::TensorView patchInput(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    const auto index = patchIndex(mesh, patch);
    return makeInputView<typename FieldType::value_type>(
        name,
        mesh.lookupObject<FieldType>(name).boundaryField()[index],
        exchangeIndex,
        physicalTime);
}

template<class FieldType>
foamnordic::fjord::MutableTensorView patchOutput(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    const auto index = patchIndex(mesh, patch);
    return makeOutputView<typename FieldType::value_type>(
        name,
        mesh.lookupObjectRef<FieldType>(name).boundaryFieldRef()[index],
        exchangeIndex,
        physicalTime);
}

[[noreturn]] void unsupported(const word& name) {
    throw std::runtime_error(
        "FoamNordic field is missing or has an unsupported volume type: "
        + std::string(name.c_str()));
}

struct ComponentSelection final {
    word field;
    direction component;
};

std::optional<ComponentSelection> componentSelection(const word& name) {
    const std::string value(name.c_str());
    if (value.size() < 3 || value[value.size() - 2] != '.') {
        return std::nullopt;
    }
    const auto suffix = value.back();
    if (suffix != 'x' && suffix != 'y' && suffix != 'z') {
        return std::nullopt;
    }
    return ComponentSelection{
        word(value.substr(0, value.size() - 2)),
        static_cast<direction>(suffix == 'x' ? 0 : (suffix == 'y' ? 1 : 2)),
    };
}

}  // namespace

#define FOAMNORDIC_DISPATCH_FIELD(ACTION)                                      \
    if (contains<volScalarField>(mesh, name)) {                                \
        return ACTION<volScalarField>(mesh, name, exchangeIndex, physicalTime);\
    }                                                                          \
    if (contains<volVectorField>(mesh, name)) {                                \
        return ACTION<volVectorField>(mesh, name, exchangeIndex, physicalTime);\
    }                                                                          \
    if (contains<volSphericalTensorField>(mesh, name)) {                       \
        return ACTION<volSphericalTensorField>(                                \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<volSymmTensorField>(mesh, name)) {                            \
        return ACTION<volSymmTensorField>(                                     \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<volTensorField>(mesh, name)) {                                \
        return ACTION<volTensorField>(mesh, name, exchangeIndex, physicalTime);\
    }                                                                          \
    if (contains<surfaceScalarField>(mesh, name)) {                            \
        return ACTION<surfaceScalarField>(                                     \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<surfaceVectorField>(mesh, name)) {                            \
        return ACTION<surfaceVectorField>(                                     \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<surfaceSphericalTensorField>(mesh, name)) {                   \
        return ACTION<surfaceSphericalTensorField>(                            \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<surfaceSymmTensorField>(mesh, name)) {                        \
        return ACTION<surfaceSymmTensorField>(                                 \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    if (contains<surfaceTensorField>(mesh, name)) {                            \
        return ACTION<surfaceTensorField>(                                     \
            mesh, name, exchangeIndex, physicalTime);                          \
    }                                                                          \
    unsupported(name)

foamnordic::fjord::TensorView inputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime,
    scalarField* scratch) {
    if (name == "x" || name == "y" || name == "z") {
        if (scratch == nullptr) {
            throw std::invalid_argument(
                "FoamNordic coordinate input requires exchange-owned storage");
        }
        const direction component = name == "x" ? 0 : (name == "y" ? 1 : 2);
        *scratch = mesh.C().primitiveField().component(component);
        return makeInputView(
            name, *scratch, exchangeIndex, physicalTime);
    }
    if (const auto selected = componentSelection(name)) {
        if (scratch == nullptr) {
            throw std::invalid_argument(
                "FoamNordic component input requires exchange-owned storage");
        }
        if (!contains<volVectorField>(mesh, selected->field)) {
            unsupported(name);
        }
        *scratch = mesh.lookupObject<volVectorField>(selected->field)
                       .primitiveField()
                       .component(selected->component);
        return makeInputView(name, *scratch, exchangeIndex, physicalTime);
    }
    FOAMNORDIC_DISPATCH_FIELD(input);
}

foamnordic::fjord::MutableTensorView outputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime,
    scalarField* scratch) {
    if (const auto selected = componentSelection(name)) {
        if (scratch == nullptr) {
            throw std::invalid_argument(
                "FoamNordic component output requires exchange-owned storage");
        }
        if (!contains<volVectorField>(mesh, selected->field)) {
            unsupported(name);
        }
        *scratch = mesh.lookupObject<volVectorField>(selected->field)
                       .primitiveField()
                       .component(selected->component);
        return makeOutputView(name, *scratch, exchangeIndex, physicalTime);
    }
    FOAMNORDIC_DISPATCH_FIELD(output);
}

#undef FOAMNORDIC_DISPATCH_FIELD

#define FOAMNORDIC_DISPATCH_PATCH(ACTION)                                    \
    if (contains<volScalarField>(mesh, name)) {                               \
        return ACTION<volScalarField>(                                       \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<volVectorField>(mesh, name)) {                               \
        return ACTION<volVectorField>(                                       \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<volSphericalTensorField>(mesh, name)) {                      \
        return ACTION<volSphericalTensorField>(                              \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<volSymmTensorField>(mesh, name)) {                           \
        return ACTION<volSymmTensorField>(                                   \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<volTensorField>(mesh, name)) {                               \
        return ACTION<volTensorField>(                                       \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<surfaceScalarField>(mesh, name)) {                           \
        return ACTION<surfaceScalarField>(                                   \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<surfaceVectorField>(mesh, name)) {                           \
        return ACTION<surfaceVectorField>(                                   \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<surfaceSphericalTensorField>(mesh, name)) {                  \
        return ACTION<surfaceSphericalTensorField>(                          \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<surfaceSymmTensorField>(mesh, name)) {                       \
        return ACTION<surfaceSymmTensorField>(                               \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    if (contains<surfaceTensorField>(mesh, name)) {                           \
        return ACTION<surfaceTensorField>(                                   \
            mesh, name, patch, exchangeIndex, physicalTime);                 \
    }                                                                         \
    unsupported(name)

foamnordic::fjord::TensorView inputPatchView(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    FOAMNORDIC_DISPATCH_PATCH(patchInput);
}

foamnordic::fjord::MutableTensorView outputPatchView(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime) {
    FOAMNORDIC_DISPATCH_PATCH(patchOutput);
}

#undef FOAMNORDIC_DISPATCH_PATCH

void correctFieldBoundary(const fvMesh& mesh, const word& name) {
    if (const auto selected = componentSelection(name)) {
        correct<volVectorField>(mesh, selected->field);
        return;
    }
#define FOAMNORDIC_CORRECT_FIELD(FIELD_TYPE)     \
    if (contains<FIELD_TYPE>(mesh, name)) {      \
        correct<FIELD_TYPE>(mesh, name);         \
        return;                                  \
    }
    FOAMNORDIC_CORRECT_FIELD(volScalarField)
    FOAMNORDIC_CORRECT_FIELD(volVectorField)
    FOAMNORDIC_CORRECT_FIELD(volSphericalTensorField)
    FOAMNORDIC_CORRECT_FIELD(volSymmTensorField)
    FOAMNORDIC_CORRECT_FIELD(volTensorField)
    FOAMNORDIC_CORRECT_FIELD(surfaceScalarField)
    FOAMNORDIC_CORRECT_FIELD(surfaceVectorField)
    FOAMNORDIC_CORRECT_FIELD(surfaceSphericalTensorField)
    FOAMNORDIC_CORRECT_FIELD(surfaceSymmTensorField)
    FOAMNORDIC_CORRECT_FIELD(surfaceTensorField)
#undef FOAMNORDIC_CORRECT_FIELD
    unsupported(name);
}

void commitOutputField(
    const fvMesh& mesh,
    const word& name,
    const scalarField* scratch) {
    const auto selected = componentSelection(name);
    if (!selected) {
        return;
    }
    if (scratch == nullptr || !contains<volVectorField>(mesh, selected->field)) {
        unsupported(name);
    }
    mesh.lookupObjectRef<volVectorField>(selected->field)
        .primitiveFieldRef()
        .replace(selected->component, *scratch);
}

}  // namespace Foam::foamNordic
