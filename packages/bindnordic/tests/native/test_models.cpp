#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "bindnordic/affine.hpp"
#include "bindnordic/manifest.hpp"

namespace {

std::vector<std::byte> bytes(const std::vector<double>& values) {
    const auto source = std::as_bytes(std::span(values));
    return {source.begin(), source.end()};
}

double value(const std::vector<std::byte>& values, std::size_t index) {
    double result{};
    std::memcpy(&result, values.data() + index * sizeof(double), sizeof(double));
    return result;
}

}  // namespace

int main() {
    bindnordic::ProgramContract contract{
        "affine",
        {{"x", bindnordic::Element::float64, 2}},
        {{"y", bindnordic::Element::float64, 1}},
    };
    contract.validate();

    auto scaler = bindnordic::AffineScaler::standard({1.0, 2.0}, {2.0, 4.0});
    auto scaled = bytes({3.0, 6.0});
    scaler.transform({"x", bindnordic::Element::float64, {1, 2}, scaled});
    assert(std::abs(value(scaled, 0) - 1.0) < 1.0e-12);
    assert(std::abs(value(scaled, 1) - 1.0) < 1.0e-12);

    bindnordic::DenseAffineKernel kernel(2, 1, {2.0, -1.0}, {0.5});
    const auto input = bytes({3.0, 4.0, 5.0, 1.0});
    const auto output = kernel.evaluate(
        {"x", bindnordic::Element::float64, {2, 2}, input},
        7,
        0.25);
    assert(std::abs(value(output.bytes, 0) - 2.5) < 1.0e-12);
    assert(std::abs(value(output.bytes, 1) - 9.5) < 1.0e-12);

    bindnordic::ModelArtifact artifact{
        bindnordic::ModelFormat::compiled,
        "model.cpp",
        contract,
        {},
        scaler,
        std::nullopt,
        "compiled",
    };
    const auto manifest = bindnordic::encode_manifest(artifact);
    const auto decoded = bindnordic::decode_manifest(manifest);
    assert(decoded.contract.name == "affine");
    assert(decoded.input_scaler.has_value());
    assert(decoded.runtime == "compiled");

    return 0;
}
