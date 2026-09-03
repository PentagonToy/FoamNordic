#include "artifact_bindings.hpp"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/manifest.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Field = std::pair<std::string, std::uint64_t>;
using Leaf = std::tuple<
    std::string,
    std::string,
    std::vector<std::uint64_t>,
    std::uint64_t,
    std::uint64_t>;

std::size_t feature_count(const std::vector<Field>& fields) {
    std::size_t result = 0;
    for (const auto& [_, components] : fields) {
        result += components;
    }
    return result;
}

std::vector<double> values(
    nb::handle scaler,
    const char* attribute,
    std::size_t features,
    double fallback) {
    if (!nb::hasattr(scaler, attribute)) {
        throw std::invalid_argument(
            std::string("Fitted scikit-learn scaler is missing ") + attribute);
    }
    nb::object value = nb::borrow<nb::object>(scaler).attr(attribute);
    if (value.is_none()) {
        return std::vector<double>(features, fallback);
    }
    if (nb::hasattr(value, "tolist")) {
        value = value.attr("tolist")();
    }
    return nb::cast<std::vector<double>>(value);
}

bool close(double actual, double expected) {
    constexpr double relative = 1.0e-9;
    constexpr double absolute = 1.0e-11;
    return std::abs(actual - expected)
        <= absolute + relative * std::abs(expected);
}

foamnordic::inference::AffineScaler function_transformer(
    nb::handle scaler,
    std::size_t features) {
    if (!nb::hasattr(scaler, "n_features_in_")
        || nb::cast<std::size_t>(scaler.attr("n_features_in_")) != features) {
        throw std::invalid_argument(
            "FunctionTransformer must be fitted for the model feature count.");
    }

    const std::size_t rows = features + 3;
    std::vector<double> flat(rows * features, 0.0);
    for (std::size_t feature = 0; feature < features; ++feature) {
        flat[(feature + 1) * features + feature] = 1.0;
        flat[(features + 1) * features + feature] =
            -1.5 + 4.0 * static_cast<double>(feature)
                / static_cast<double>(features > 1 ? features - 1 : 1);
        flat[(features + 2) * features + feature] =
            3.0 - 5.0 * static_cast<double>(feature)
                / static_cast<double>(features > 1 ? features - 1 : 1);
    }

    auto numpy = nb::module_::import_("numpy");
    auto probes = numpy.attr("asarray")(flat).attr("reshape")(rows, features);
    nb::object transformed;
    try {
        transformed = numpy.attr("asarray")(scaler.attr("transform")(probes));
    } catch (const nb::python_error&) {
        throw std::invalid_argument(
            "FunctionTransformer could not transform numeric features.");
    }
    if (nb::cast<std::size_t>(transformed.attr("ndim")) != 2) {
        throw std::invalid_argument(
            "FunctionTransformer must preserve the feature shape.");
    }
    const auto shape = nb::cast<std::pair<std::size_t, std::size_t>>(
        transformed.attr("shape"));
    if (shape.first != rows || shape.second != features) {
        throw std::invalid_argument(
            "FunctionTransformer must preserve the feature shape.");
    }
    const auto result = nb::cast<std::vector<std::vector<double>>>(
        transformed.attr("tolist")());
    const auto& bias = result.front();
    std::vector<double> gain(features);
    for (std::size_t row = 0; row < features; ++row) {
        for (std::size_t column = 0; column < features; ++column) {
            const double coefficient = result[row + 1][column] - bias[column];
            if (row == column) {
                gain[column] = coefficient;
            } else if (!close(coefficient, 0.0)) {
                throw std::invalid_argument(
                    "FunctionTransformer mixes features and cannot be represented natively.");
            }
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t feature = 0; feature < features; ++feature) {
            const double expected = flat[row * features + feature] * gain[feature]
                + bias[feature];
            if (!close(result[row][feature], expected)) {
                throw std::invalid_argument(
                    "FunctionTransformer is nonlinear and cannot be represented natively.");
            }
        }
    }
    for (const auto coefficient : gain) {
        if (!std::isfinite(coefficient) || coefficient == 0.0) {
            throw std::invalid_argument(
                "FunctionTransformer has a non-invertible feature scale.");
        }
    }
    return foamnordic::inference::AffineScaler::function(
        std::move(gain), bias);
}

std::optional<foamnordic::inference::AffineScaler> create_cpp_scaler(
    nb::object scaler,
    std::size_t features) {
    if (scaler.is_none()) {
        return std::nullopt;
    }
    const auto type_name = nb::cast<std::string>(
        scaler.attr("__class__").attr("__name__"));
    if (type_name == "StandardScaler") {
        return foamnordic::inference::AffineScaler::standard(
            values(scaler, "mean_", features, 0.0),
            values(scaler, "scale_", features, 1.0));
    }
    if (type_name == "MinMaxScaler") {
        std::optional<double> lower;
        std::optional<double> upper;
        if (nb::cast<bool>(scaler.attr("clip"))) {
            const auto range = nb::cast<std::pair<double, double>>(
                scaler.attr("feature_range"));
            lower = range.first;
            upper = range.second;
        }
        return foamnordic::inference::AffineScaler::minmax(
            values(scaler, "scale_", features, 1.0),
            values(scaler, "min_", features, 0.0),
            lower,
            upper);
    }
    if (type_name == "MaxAbsScaler") {
        return foamnordic::inference::AffineScaler::maxabs(
            values(scaler, "scale_", features, 1.0));
    }
    if (type_name == "RobustScaler") {
        return foamnordic::inference::AffineScaler::robust(
            values(scaler, "center_", features, 0.0),
            values(scaler, "scale_", features, 1.0));
    }
    if (type_name == "FunctionTransformer") {
        return function_transformer(scaler, features);
    }
    throw std::invalid_argument(
        "Unsupported fitted scikit-learn scaler: " + type_name
        + ". Expected StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, "
          "or an affine FunctionTransformer.");
}

foamnordic::fjord::Element element(const std::string& dtype) {
    if (dtype == "float32") {
        return foamnordic::fjord::Element::float32;
    }
    if (dtype == "float64") {
        return foamnordic::fjord::Element::float64;
    }
    if (dtype == "int32") {
        return foamnordic::fjord::Element::int32;
    }
    if (dtype == "int64") {
        return foamnordic::fjord::Element::int64;
    }
    throw std::invalid_argument(
        "dtype must be float32, float64, int32, or int64");
}

foamnordic::inference::ModelFormat format(const std::string& value) {
    if (value == "compiled") {
        return foamnordic::inference::ModelFormat::compiled;
    }
    if (value == "equinox") {
        return foamnordic::inference::ModelFormat::equinox;
    }
    if (value == "joblib") {
        return foamnordic::inference::ModelFormat::joblib;
    }
    if (value == "onnx") {
        return foamnordic::inference::ModelFormat::onnx;
    }
    throw std::invalid_argument(
        "model format must be compiled, equinox, joblib, or onnx");
}

std::vector<foamnordic::inference::FieldContract> fields(
    const std::vector<Field>& specifications,
    foamnordic::fjord::Element dtype) {
    std::vector<foamnordic::inference::FieldContract> result;
    result.reserve(specifications.size());
    for (const auto& [name, components] : specifications) {
        result.push_back({name, dtype, components});
    }
    return result;
}

std::vector<foamnordic::inference::TreeLeaf> leaves(
    const std::vector<Leaf>& specifications) {
    std::vector<foamnordic::inference::TreeLeaf> result;
    result.reserve(specifications.size());
    for (const auto& [path, dtype, shape, offset, count] : specifications) {
        result.push_back({path, element(dtype), shape, offset, count});
    }
    return result;
}

nb::dict manifest_dict(const foamnordic::inference::ModelArtifact& artifact) {
    nb::dict result;
    result["schema_version"] = artifact.schema_version;
    result["format"] = foamnordic::inference::name(artifact.format);
    result["artifact_path"] = artifact.artifact_path;
    result["name"] = artifact.contract.name;
    nb::list inputs;
    for (const auto& field : artifact.contract.inputs) {
        inputs.append(nb::make_tuple(
            field.name,
            field.components,
            field.element == foamnordic::fjord::Element::float32
                ? "float32"
                : "float64"));
    }
    nb::list outputs;
    for (const auto& field : artifact.contract.outputs) {
        outputs.append(nb::make_tuple(
            field.name,
            field.components,
            field.element == foamnordic::fjord::Element::float32
                ? "float32"
                : "float64"));
    }
    result["inputs"] = std::move(inputs);
    result["outputs"] = std::move(outputs);
    const auto scaler = [](const std::optional<foamnordic::inference::AffineScaler>& value) {
        if (!value) {
            return nb::object(nb::none());
        }
        nb::dict result;
        result["kind"] = foamnordic::inference::name(value->kind());
        result["gain"] = nb::cast(value->gain());
        result["bias"] = nb::cast(value->bias());
        result["clip_lower"] = value->clip_lower()
                                   ? nb::cast(*value->clip_lower())
                                   : nb::object(nb::none());
        result["clip_upper"] = value->clip_upper()
                                   ? nb::cast(*value->clip_upper())
                                   : nb::object(nb::none());
        return nb::object(std::move(result));
    };
    result["input_scaler"] = scaler(artifact.input_scaler);
    result["output_scaler"] = scaler(artifact.output_scaler);
    result["runtime"] = artifact.runtime.empty()
                            ? nb::object(nb::none())
                            : nb::cast(artifact.runtime);
    return result;
}

foamnordic::inference::ModelArtifact make_artifact(
    const std::string& artifact_path,
    const std::string& name,
    const std::string& model_format,
    const std::vector<Field>& inputs,
    const std::vector<Field>& outputs,
    const std::string& dtype,
    const std::vector<Leaf>& tree_leaves,
    nb::object x_scaler,
    nb::object y_scaler,
    const std::string& runtime) {
    const auto value_type = element(dtype);
    return {
        runtime.empty() ? 1U : 2U,
        format(model_format),
        artifact_path,
        {name, fields(inputs, value_type), fields(outputs, value_type)},
        leaves(tree_leaves),
        create_cpp_scaler(std::move(x_scaler), feature_count(inputs)),
        create_cpp_scaler(std::move(y_scaler), feature_count(outputs)),
        runtime,
    };
}

}  // namespace

void bind_artifacts(nb::module_& module) {
    module.def(
        "write_model_manifest",
        [](const std::string& manifest_path,
           const std::string& artifact_path,
           const std::string& name,
           const std::string& model_format,
           const std::vector<Field>& inputs,
           const std::vector<Field>& outputs,
           const std::string& dtype,
           const std::vector<Leaf>& tree_leaves,
           nb::object x_scaler,
           nb::object y_scaler,
           const std::string& runtime) {
            foamnordic::inference::write_manifest(
                manifest_path,
                make_artifact(
                    artifact_path,
                    name,
                    model_format,
                    inputs,
                    outputs,
                    dtype,
                    tree_leaves,
                    std::move(x_scaler),
                    std::move(y_scaler),
                    runtime));
        },
        "manifest_path"_a,
        "artifact_path"_a,
        "name"_a,
        "format"_a,
        "inputs"_a,
        "outputs"_a,
        "dtype"_a = "float64",
        "tree_leaves"_a = std::vector<Leaf>{},
        "x_scaler"_a = nb::none(),
        "y_scaler"_a = nb::none(),
        "runtime"_a = "",
        "Write one backend-neutral FNOM manifest.");

    module.def(
        "write_model_bundle",
        [](const std::string& manifest_path,
           const std::string& payload_path,
           const std::string& name,
           const std::string& model_format,
           const std::vector<Field>& inputs,
           const std::vector<Field>& outputs,
           const std::string& dtype,
           const std::vector<Leaf>& tree_leaves,
           nb::object x_scaler,
           nb::object y_scaler,
           const std::string& runtime) {
            foamnordic::inference::write_bundle(
                manifest_path,
                make_artifact(
                    std::filesystem::path(payload_path).filename().string(),
                    name,
                    model_format,
                    inputs,
                    outputs,
                    dtype,
                    tree_leaves,
                    std::move(x_scaler),
                    std::move(y_scaler),
                    runtime),
                payload_path);
        },
        "manifest_path"_a,
        "payload_path"_a,
        "name"_a,
        "format"_a,
        "inputs"_a,
        "outputs"_a,
        "dtype"_a = "float64",
        "tree_leaves"_a = std::vector<Leaf>{},
        "x_scaler"_a = nb::none(),
        "y_scaler"_a = nb::none(),
        "runtime"_a = "",
        "Write one self-contained, uncompressed FNOM model bundle.");

    module.def(
        "read_model_manifest",
        [](const std::string& path) {
            auto result = manifest_dict(foamnordic::inference::read_manifest(path));
            result["bundled"] = foamnordic::inference::is_bundle(path);
            return result;
        },
        "path"_a,
        "Read the launch metadata required by a managed model worker.");

    module.def(
        "read_model_payload",
        [](const std::string& path) {
            const auto payload = foamnordic::inference::read_bundle_payload(path);
            return nb::bytes(
                reinterpret_cast<const char*>(payload.data()), payload.size());
        },
        "path"_a,
        "Read the embedded payload from a self-contained FNOM bundle.");

    module.def(
        "extract_model_payload",
        [](const std::string& path, const std::string& destination) {
            foamnordic::inference::extract_bundle_payload(path, destination);
        },
        "path"_a,
        "destination"_a,
        "Stream an embedded FNOM payload into a worker-local file.");

    module.def(
        "read_model_payload_region",
        [](const std::string& path) {
            const auto region = foamnordic::inference::bundle_payload_region(path);
            return std::make_pair(region.offset, region.size);
        },
        "path"_a,
        "Read the byte offset and size of an embedded FNOM payload.");
}
