// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * BindNordic
 */
// clang-format on

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "bindnordic/affine.hpp"
#include "bindnordic/manifest.hpp"
#include "bindnordic/table.hpp"

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

template<class Value>
void append(std::vector<std::byte>& destination, Value value) {
    const auto source = std::as_bytes(std::span(&value, 1));
    destination.insert(destination.end(), source.begin(), source.end());
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
    const auto kernel_capabilities = kernel.capabilities();
    assert(kernel_capabilities.valid());
    assert(kernel_capabilities.input_memory == bindnordic::MemorySpace::host);
    assert(kernel_capabilities.output_memory == bindnordic::MemorySpace::host);
    assert(
        kernel_capabilities.completion
        == bindnordic::CompletionMode::synchronous);
    assert(!kernel_capabilities.persistent_device_state);
    assert(kernel_capabilities.alignment_bytes == 1);
    auto invalid_capabilities = kernel_capabilities;
    invalid_capabilities.alignment_bytes = 3;
    assert(!invalid_capabilities.valid());
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
        "numpy",
    };
    const auto manifest = bindnordic::encode_manifest(artifact);
    const auto decoded = bindnordic::decode_manifest(manifest);
    assert(decoded.contract.name == "affine");
    assert(decoded.input_scaler.has_value());
    assert(decoded.runtime == "compiled");

    std::vector<std::byte> table_payload;
    for (const char character : std::string("FNOTAB1\0", 8)) {
        append(table_payload, character);
    }
    append(table_payload, std::uint8_t{2});
    append(table_payload, std::uint8_t{2});
    append(table_payload, std::uint16_t{0});
    append(table_payload, std::uint64_t{1});
    for (const auto& axis : std::array<std::vector<double>, 2>{
             std::vector<double>{0.0, 1.0},
             std::vector<double>{0.0, 2.0}}) {
        append(table_payload, std::uint8_t{0});
        append(table_payload, std::uint64_t{axis.size()});
        for (const auto point : axis) append(table_payload, point);
    }
    for (const auto point : std::vector<double>{0.0, 4.0, 2.0, 6.0}) {
        append(table_payload, point);
    }
    bindnordic::TableKernel table(table_payload, 2, 1, bindnordic::Element::float64);
    const auto table_input = bytes({0.25, 0.5, 2.0, -1.0});
    const auto table_output = table.evaluate(
        {"coordinates", bindnordic::Element::float64, {2, 2}, table_input},
        0,
        0.0);
    assert(std::abs(value(table_output.bytes, 0) - 1.5) < 1.0e-12);
    assert(std::abs(value(table_output.bytes, 1) - 2.0) < 1.0e-12);

    std::vector<std::byte> float_table_payload;
    for (const char character : std::string("FNOTAB1\0", 8)) {
        append(float_table_payload, character);
    }
    append(float_table_payload, std::uint8_t{1});
    append(float_table_payload, std::uint8_t{1});
    append(float_table_payload, std::uint16_t{0});
    append(float_table_payload, std::uint64_t{1});
    append(float_table_payload, std::uint8_t{0});
    append(float_table_payload, std::uint64_t{2});
    append(float_table_payload, 0.0);
    append(float_table_payload, 1.0);
    append(float_table_payload, 1.0F);
    append(float_table_payload, 3.0F);
    bindnordic::TableKernel float_table(
        float_table_payload, 1, 1, bindnordic::Element::float32);
    const std::vector<float> float_coordinates{0.25F};
    const auto float_output = float_table.evaluate(
        {
            "coordinates",
            bindnordic::Element::float32,
            {1, 1},
            bindnordic::as_bytes(std::span(float_coordinates)),
        },
        0,
        0.0);
    float interpolated{};
    std::memcpy(&interpolated, float_output.bytes.data(), sizeof(float));
    assert(std::abs(interpolated - 1.5F) < 1.0e-6F);

    std::vector<std::byte> uniform_table_payload;
    for (const char character : std::string("FNOTAB2\0", 8)) {
        append(uniform_table_payload, character);
    }
    append(uniform_table_payload, std::uint8_t{2});
    append(uniform_table_payload, std::uint8_t{1});
    append(uniform_table_payload, std::uint16_t{0});
    append(uniform_table_payload, std::uint64_t{1});
    append(uniform_table_payload, std::uint8_t{0});
    append(uniform_table_payload, std::uint8_t{1});
    append(uniform_table_payload, std::uint16_t{0});
    append(uniform_table_payload, std::uint64_t{3});
    append(uniform_table_payload, 0.0);
    append(uniform_table_payload, 0.5);
    for (const auto point : std::vector<double>{1.0, 2.0, 5.0}) {
        append(uniform_table_payload, point);
    }
    bindnordic::TableKernel uniform_table(
        uniform_table_payload, 1, 1, bindnordic::Element::float64);
    const auto uniform_output = uniform_table.evaluate(
        {"coordinates", bindnordic::Element::float64, {1, 1}, bytes({0.75})},
        0,
        0.0);
    assert(std::abs(value(uniform_output.bytes, 0) - 3.5) < 1.0e-12);

    std::vector<std::byte> collapsed_axis_payload;
    for (const char character : std::string("FNOTAB2\0", 8)) {
        append(collapsed_axis_payload, character);
    }
    append(collapsed_axis_payload, std::uint8_t{2});
    append(collapsed_axis_payload, std::uint8_t{1});
    append(collapsed_axis_payload, std::uint16_t{0});
    append(collapsed_axis_payload, std::uint64_t{1});
    append(collapsed_axis_payload, std::uint8_t{0});
    append(collapsed_axis_payload, std::uint8_t{1});
    append(collapsed_axis_payload, std::uint16_t{0});
    append(collapsed_axis_payload, std::uint64_t{2});
    append(collapsed_axis_payload, 1.0e16);
    append(collapsed_axis_payload, 1.0);
    append(collapsed_axis_payload, 0.0);
    append(collapsed_axis_payload, 1.0);
    bool rejected_collapsed_axis = false;
    try {
        bindnordic::TableKernel collapsed_axis(
            collapsed_axis_payload, 1, 1, bindnordic::Element::float64);
    } catch (const std::invalid_argument&) {
        rejected_collapsed_axis = true;
    }
    assert(rejected_collapsed_axis);

    return 0;
}
