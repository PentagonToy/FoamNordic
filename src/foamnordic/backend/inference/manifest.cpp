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

#include "foamnordic/backend/inference/manifest.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace foamnordic::closure {
namespace {

constexpr std::array<std::byte, 8> magic{
    std::byte{'F'}, std::byte{'N'}, std::byte{'O'}, std::byte{'M'},
    std::byte{'A'}, std::byte{'N'}, std::byte{'1'}, std::byte{0},
};
constexpr std::uint32_t maximum_entries = 65536;
constexpr std::uint32_t maximum_string_bytes = 1024 * 1024;
constexpr std::uint64_t maximum_manifest_bytes = 64ULL * 1024ULL * 1024ULL;

class Writer {
public:
    void raw(std::span<const std::byte> bytes) {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            u8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            u8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
        }
    }

    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void string(const std::string& value) {
        if (value.size() > maximum_string_bytes) {
            throw std::invalid_argument("Manifest string exceeds its size limit.");
        }
        u32(static_cast<std::uint32_t>(value.size()));
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }

    [[nodiscard]] std::vector<std::byte> finish() && { return std::move(bytes_); }

private:
    std::vector<std::byte> bytes_;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::span<const std::byte> raw(std::size_t count) {
        if (count > bytes_.size() - offset_) {
            throw std::invalid_argument("Manifest ended before a complete value.");
        }
        const auto result = bytes_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

    [[nodiscard]] std::uint8_t u8() {
        return std::to_integer<std::uint8_t>(raw(1).front());
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] double f64() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::string string() {
        const auto count = u32();
        if (count > maximum_string_bytes) {
            throw std::invalid_argument("Manifest string exceeds its size limit.");
        }
        const auto value = raw(count);
        return {
            reinterpret_cast<const char*>(value.data()),
            value.size(),
        };
    }

    [[nodiscard]] bool empty() const noexcept { return offset_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t offset_{0};
};

std::uint8_t encode_element(fjord::Element element) {
    if (fjord::element_size(element) == 0) {
        throw std::invalid_argument("Manifest contains an unsupported element type.");
    }
    return static_cast<std::uint8_t>(element);
}

fjord::Element decode_element(std::uint8_t value) {
    const auto element = static_cast<fjord::Element>(value);
    if (fjord::element_size(element) == 0) {
        throw std::invalid_argument("Manifest element type is invalid.");
    }
    return element;
}

void write_fields(Writer& writer, const std::vector<FieldContract>& fields) {
    if (fields.size() > maximum_entries) {
        throw std::invalid_argument("Manifest contains too many fields.");
    }
    writer.u32(static_cast<std::uint32_t>(fields.size()));
    for (const auto& field : fields) {
        writer.string(field.name);
        writer.u8(encode_element(field.element));
        writer.u64(field.components);
    }
}

std::vector<FieldContract> read_fields(Reader& reader) {
    const auto count = reader.u32();
    if (count > maximum_entries) {
        throw std::invalid_argument("Manifest contains too many fields.");
    }
    std::vector<FieldContract> fields;
    fields.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        fields.push_back({reader.string(), decode_element(reader.u8()), reader.u64()});
    }
    return fields;
}

std::uint8_t encode_scaler_kind(ScalerKind kind) {
    switch (kind) {
        case ScalerKind::standard:
            return 1;
        case ScalerKind::minmax:
            return 2;
        case ScalerKind::maxabs:
            return 4;
        case ScalerKind::robust:
            return 3;
        case ScalerKind::function:
            return 5;
    }
    throw std::invalid_argument("Manifest scaler kind is invalid.");
}

ScalerKind decode_scaler_kind(std::uint8_t value) {
    switch (value) {
        case 1:
            return ScalerKind::standard;
        case 2:
            return ScalerKind::minmax;
        case 3:
            return ScalerKind::robust;
        case 4:
            return ScalerKind::maxabs;
        case 5:
            return ScalerKind::function;
        default:
            throw std::invalid_argument("Manifest scaler kind is invalid.");
    }
}

void write_scaler(Writer& writer, const std::optional<AffineScaler>& scaler) {
    writer.u8(scaler ? 1 : 0);
    if (!scaler) {
        return;
    }
    if (scaler->features() > maximum_entries) {
        throw std::invalid_argument("Manifest scaler contains too many features.");
    }
    writer.u8(encode_scaler_kind(scaler->kind()));
    writer.u32(static_cast<std::uint32_t>(scaler->features()));
    for (const auto value : scaler->gain()) {
        writer.f64(value);
    }
    for (const auto value : scaler->bias()) {
        writer.f64(value);
    }
    const auto clipped = scaler->clip_lower().has_value();
    writer.u8(clipped ? 1 : 0);
    if (clipped) {
        writer.f64(*scaler->clip_lower());
        writer.f64(*scaler->clip_upper());
    }
}

std::optional<AffineScaler> read_scaler(Reader& reader) {
    const auto present = reader.u8();
    if (present > 1) {
        throw std::invalid_argument("Manifest scaler presence flag is invalid.");
    }
    if (present == 0) {
        return std::nullopt;
    }
    const auto kind = decode_scaler_kind(reader.u8());
    const auto count = reader.u32();
    if (count == 0 || count > maximum_entries) {
        throw std::invalid_argument("Manifest scaler feature count is invalid.");
    }
    std::vector<double> gain(count);
    std::vector<double> bias(count);
    for (auto& value : gain) {
        value = reader.f64();
    }
    for (auto& value : bias) {
        value = reader.f64();
    }
    const auto clipped = reader.u8();
    if (clipped > 1) {
        throw std::invalid_argument("Manifest scaler clipping flag is invalid.");
    }
    std::optional<double> lower;
    std::optional<double> upper;
    if (clipped == 1) {
        lower = reader.f64();
        upper = reader.f64();
    }
    return AffineScaler(kind, std::move(gain), std::move(bias), lower, upper);
}

std::uint8_t encode_format(ModelFormat format) {
    switch (format) {
        case ModelFormat::equinox:
            return 1;
        case ModelFormat::joblib:
            return 2;
        case ModelFormat::onnx:
            return 3;
    }
    throw std::invalid_argument("Manifest model format is invalid.");
}

ModelFormat decode_format(std::uint8_t value) {
    switch (value) {
        case 1:
            return ModelFormat::equinox;
        case 2:
            return ModelFormat::joblib;
        case 3:
            return ModelFormat::onnx;
        default:
            throw std::invalid_argument("Manifest model format is invalid.");
    }
}

}  // namespace

std::vector<std::byte> encode_manifest(const ModelArtifact& artifact) {
    artifact.validate();
    Writer writer;
    writer.raw(magic);
    writer.u32(artifact.schema_version);
    writer.u8(encode_format(artifact.format));
    writer.string(artifact.artifact_path);
    writer.string(artifact.contract.name);
    write_fields(writer, artifact.contract.inputs);
    write_fields(writer, artifact.contract.outputs);
    write_scaler(writer, artifact.input_scaler);
    write_scaler(writer, artifact.output_scaler);
    if (artifact.tree_leaves.size() > maximum_entries) {
        throw std::invalid_argument("Manifest contains too many tree leaves.");
    }
    writer.u32(static_cast<std::uint32_t>(artifact.tree_leaves.size()));
    for (const auto& leaf : artifact.tree_leaves) {
        if (leaf.shape.size() > maximum_entries) {
            throw std::invalid_argument("Manifest tree leaf rank is too large.");
        }
        writer.string(leaf.path);
        writer.u8(encode_element(leaf.element));
        writer.u32(static_cast<std::uint32_t>(leaf.shape.size()));
        for (const auto extent : leaf.shape) {
            writer.u64(extent);
        }
        writer.u64(leaf.byte_offset);
        writer.u64(leaf.byte_count);
    }
    return std::move(writer).finish();
}

ModelArtifact decode_manifest(std::span<const std::byte> bytes) {
    if (bytes.size() > maximum_manifest_bytes) {
        throw std::invalid_argument("Manifest exceeds its size limit.");
    }
    Reader reader(bytes);
    if (!std::ranges::equal(reader.raw(magic.size()), magic)) {
        throw std::invalid_argument("Manifest magic is invalid.");
    }
    ModelArtifact artifact;
    artifact.schema_version = reader.u32();
    artifact.format = decode_format(reader.u8());
    artifact.artifact_path = reader.string();
    artifact.contract.name = reader.string();
    artifact.contract.inputs = read_fields(reader);
    artifact.contract.outputs = read_fields(reader);
    artifact.input_scaler = read_scaler(reader);
    artifact.output_scaler = read_scaler(reader);
    const auto leaf_count = reader.u32();
    if (leaf_count > maximum_entries) {
        throw std::invalid_argument("Manifest contains too many tree leaves.");
    }
    artifact.tree_leaves.reserve(leaf_count);
    for (std::uint32_t leaf_index = 0; leaf_index < leaf_count; ++leaf_index) {
        TreeLeaf leaf;
        leaf.path = reader.string();
        leaf.element = decode_element(reader.u8());
        const auto rank = reader.u32();
        if (rank == 0 || rank > maximum_entries) {
            throw std::invalid_argument("Manifest tree leaf rank is invalid.");
        }
        leaf.shape.reserve(rank);
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
            leaf.shape.push_back(reader.u64());
        }
        leaf.byte_offset = reader.u64();
        leaf.byte_count = reader.u64();
        artifact.tree_leaves.push_back(std::move(leaf));
    }
    if (!reader.empty()) {
        throw std::invalid_argument("Manifest contains trailing data.");
    }
    artifact.validate();
    return artifact;
}

void write_manifest(
    const std::filesystem::path& path,
    const ModelArtifact& artifact) {
    const auto bytes = encode_manifest(artifact);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Could not open manifest for writing: " + path.string());
    }
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("Could not write manifest: " + path.string());
    }
}

ModelArtifact read_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Could not open manifest for reading: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > maximum_manifest_bytes) {
        throw std::runtime_error("Manifest file size is invalid: " + path.string());
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!stream && !bytes.empty()) {
        throw std::runtime_error("Could not read manifest: " + path.string());
    }
    return decode_manifest(bytes);
}

}  // namespace foamnordic::closure
