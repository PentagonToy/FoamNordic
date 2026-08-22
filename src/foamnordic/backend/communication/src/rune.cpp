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

#include "foamnordic/fjord/rune.hpp"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace foamnordic::fjord {
namespace {

template <class Integer>
void append_little(std::span<std::byte> target, std::size_t offset, Integer value) {
    static_assert(std::is_unsigned_v<Integer>);
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        target[offset + index] = std::byte((value >> (index * 8U)) & 0xffU);
    }
}

template <class Integer>
[[nodiscard]] Integer read_little(std::span<const std::byte> source, std::size_t offset) {
    static_assert(std::is_unsigned_v<Integer>);
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        value |= static_cast<Integer>(std::to_integer<unsigned char>(source[offset + index]))
                 << (index * 8U);
    }
    return value;
}

void validate_prefix(const RunePrefix& prefix) {
    if (prefix.version != rune_version) {
        throw std::runtime_error("Unsupported Rune protocol version.");
    }
    switch (prefix.kind) {
        case RuneKind::hello:
        case RuneKind::hello_accept:
        case RuneKind::hello_reject:
        case RuneKind::tensor:
        case RuneKind::complete:
        case RuneKind::shutdown:
        case RuneKind::error:
        case RuneKind::shm_offer:
        case RuneKind::shm_ready:
            break;
        default:
            throw std::runtime_error("Unsupported Rune message kind.");
    }
    if (prefix.peers == 0 || prefix.rank >= prefix.peers) {
        throw std::runtime_error("Rune rank metadata is invalid.");
    }
    if (prefix.kind == RuneKind::tensor) {
        if (element_size(prefix.element) == 0) {
            throw std::runtime_error("Unsupported Rune tensor element type.");
        }
        if (prefix.dimensions == 0 || prefix.dimensions > 16) {
            throw std::runtime_error("Rune tensor dimensions must be between 1 and 16.");
        }
        if (prefix.name_bytes == 0 || prefix.name_bytes > 4096) {
            throw std::runtime_error("Rune tensor name has an invalid length.");
        }
        if (prefix.shape_bytes != static_cast<std::uint32_t>(prefix.dimensions) * 8U) {
            throw std::runtime_error("Rune tensor shape metadata has an invalid length.");
        }
    } else if ((prefix.kind != RuneKind::shm_offer && prefix.name_bytes != 0)
               || prefix.shape_bytes != 0 || prefix.payload_bytes != 0
               || prefix.dimensions != 0
               || (prefix.kind == RuneKind::shm_offer
                   && (prefix.name_bytes == 0 || prefix.name_bytes > 4096))) {
        throw std::runtime_error("Rune control messages must not contain tensor data.");
    }
}

}  // namespace

std::array<std::byte, rune_prefix_size> encode_prefix(const RunePrefix& prefix) {
    validate_prefix(prefix);
    std::array<std::byte, rune_prefix_size> bytes{};
    std::copy(rune_mark.begin(), rune_mark.end(), bytes.begin());
    append_little<std::uint16_t>(bytes, 4, prefix.version);
    bytes[6] = std::byte(static_cast<std::uint8_t>(prefix.kind));
    bytes[7] = std::byte(static_cast<std::uint8_t>(prefix.element));
    append_little<std::uint32_t>(bytes, 8, prefix.flags);
    append_little<std::uint16_t>(bytes, 12, prefix.dimensions);
    append_little<std::uint32_t>(bytes, 16, prefix.name_bytes);
    append_little<std::uint32_t>(bytes, 20, prefix.shape_bytes);
    append_little<std::uint64_t>(bytes, 24, prefix.payload_bytes);
    append_little<std::uint64_t>(bytes, 32, prefix.exchange_index);
    append_little<std::uint32_t>(bytes, 40, prefix.rank);
    append_little<std::uint32_t>(bytes, 44, prefix.peers);
    append_little<std::uint64_t>(
        bytes, 48, std::bit_cast<std::uint64_t>(prefix.physical_time));
    append_little<std::uint64_t>(bytes, 56, prefix.session_id);
    append_little<std::uint64_t>(bytes, 64, prefix.maximum_payload);
    append_little<std::uint64_t>(bytes, 72, prefix.solver_time_index);
    return bytes;
}

RunePrefix decode_prefix(std::span<const std::byte, rune_prefix_size> bytes) {
    if (!std::equal(rune_mark.begin(), rune_mark.end(), bytes.begin())) {
        throw std::runtime_error("Invalid Rune protocol marker.");
    }
    RunePrefix prefix;
    prefix.version = read_little<std::uint16_t>(bytes, 4);
    prefix.kind = static_cast<RuneKind>(std::to_integer<std::uint8_t>(bytes[6]));
    prefix.element = static_cast<Element>(std::to_integer<std::uint8_t>(bytes[7]));
    prefix.flags = read_little<std::uint32_t>(bytes, 8);
    prefix.dimensions = read_little<std::uint16_t>(bytes, 12);
    prefix.name_bytes = read_little<std::uint32_t>(bytes, 16);
    prefix.shape_bytes = read_little<std::uint32_t>(bytes, 20);
    prefix.payload_bytes = read_little<std::uint64_t>(bytes, 24);
    prefix.exchange_index = read_little<std::uint64_t>(bytes, 32);
    prefix.rank = read_little<std::uint32_t>(bytes, 40);
    prefix.peers = read_little<std::uint32_t>(bytes, 44);
    prefix.physical_time =
        std::bit_cast<double>(read_little<std::uint64_t>(bytes, 48));
    prefix.session_id = read_little<std::uint64_t>(bytes, 56);
    prefix.maximum_payload = read_little<std::uint64_t>(bytes, 64);
    prefix.solver_time_index = read_little<std::uint64_t>(bytes, 72);
    validate_prefix(prefix);
    return prefix;
}

std::vector<std::byte> encode_shape(std::span<const std::uint64_t> shape) {
    std::vector<std::byte> bytes(shape.size() * sizeof(std::uint64_t));
    for (std::size_t index = 0; index < shape.size(); ++index) {
        append_little<std::uint64_t>(bytes, index * 8, shape[index]);
    }
    return bytes;
}

std::vector<std::uint64_t> decode_shape(
    std::span<const std::byte> bytes,
    std::uint16_t dimensions) {
    if (bytes.size() != static_cast<std::size_t>(dimensions) * 8) {
        throw std::runtime_error("Rune shape byte count does not match its dimensions.");
    }
    std::vector<std::uint64_t> shape(dimensions);
    for (std::size_t index = 0; index < dimensions; ++index) {
        shape[index] = read_little<std::uint64_t>(bytes, index * 8);
    }
    return shape;
}

std::vector<std::byte> encode_tensor(const TensorView& tensor) {
    tensor.validate();
    if (tensor.shape.size() > std::numeric_limits<std::uint8_t>::max()) {
        throw std::invalid_argument("Tensor has too many dimensions for Rune.");
    }
    const auto shape = encode_shape(tensor.shape);
    const RunePrefix prefix{
        .kind = RuneKind::tensor,
        .element = tensor.element,
        .dimensions = static_cast<std::uint16_t>(tensor.shape.size()),
        .name_bytes = static_cast<std::uint32_t>(tensor.name.size()),
        .shape_bytes = static_cast<std::uint32_t>(shape.size()),
        .payload_bytes = static_cast<std::uint64_t>(tensor.bytes.size()),
        .exchange_index = tensor.time_index,
        .physical_time = tensor.physical_time,
        .solver_time_index = tensor.solver_time_index,
    };
    const auto header = encode_prefix(prefix);
    std::vector<std::byte> frame;
    frame.reserve(header.size() + tensor.name.size() + shape.size() + tensor.bytes.size());
    frame.insert(frame.end(), header.begin(), header.end());
    const auto name = std::as_bytes(std::span(tensor.name));
    frame.insert(frame.end(), name.begin(), name.end());
    frame.insert(frame.end(), shape.begin(), shape.end());
    frame.insert(frame.end(), tensor.bytes.begin(), tensor.bytes.end());
    return frame;
}

Tensor decode_tensor(std::span<const std::byte> frame) {
    if (frame.size() < rune_prefix_size) {
        throw std::runtime_error("Rune frame is shorter than its prefix.");
    }
    const auto prefix = decode_prefix(
        std::span<const std::byte, rune_prefix_size>(frame.first(rune_prefix_size)));
    if (prefix.kind != RuneKind::tensor) {
        throw std::runtime_error("Rune frame does not contain a tensor.");
    }
    const auto total = rune_prefix_size + prefix.name_bytes + prefix.shape_bytes
                       + prefix.payload_bytes;
    if (total != frame.size()) {
        throw std::runtime_error("Rune frame length does not match its prefix.");
    }
    std::size_t offset = rune_prefix_size;
    std::string name(prefix.name_bytes, '\0');
    std::memcpy(name.data(), frame.data() + offset, prefix.name_bytes);
    offset += prefix.name_bytes;
    auto shape = decode_shape(frame.subspan(offset, prefix.shape_bytes), prefix.dimensions);
    offset += prefix.shape_bytes;
    std::vector<std::byte> payload(prefix.payload_bytes);
    std::copy(frame.begin() + static_cast<std::ptrdiff_t>(offset), frame.end(), payload.begin());
    Tensor tensor{
        name,
        prefix.element,
        std::move(shape),
        std::move(payload),
        prefix.exchange_index,
        prefix.physical_time,
        prefix.solver_time_index};
    tensor.view().validate();
    return tensor;
}

}  // namespace foamnordic::fjord
