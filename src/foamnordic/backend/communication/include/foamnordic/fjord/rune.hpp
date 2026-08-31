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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "foamnordic/fjord/tensor.hpp"

namespace foamnordic::fjord {

inline constexpr std::array<std::byte, 4> rune_mark{
    std::byte{'F'}, std::byte{'N'}, std::byte{'R'}, std::byte{'D'}};
inline constexpr std::uint16_t rune_version = 2;
inline constexpr std::size_t rune_prefix_size = 80;

enum class RuneKind : std::uint8_t {
    hello = 1,
    hello_accept = 2,
    hello_reject = 3,
    tensor = 4,
    complete = 5,
    shutdown = 6,
    error = 7,
    shm_offer = 8,
    shm_ready = 9,
    ucx_offer = 10,
    ucx_ready = 11,
};

enum class Capability : std::uint32_t {
    none = 0,
    uds = 1U << 0U,
    shm = 1U << 1U,
    tcp = 1U << 2U,
    ucx = 1U << 3U,
};

[[nodiscard]] constexpr Capability operator|(Capability left, Capability right) {
    return static_cast<Capability>(
        static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr Capability operator&(Capability left, Capability right) {
    return static_cast<Capability>(
        static_cast<std::uint32_t>(left) & static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr bool any(Capability capabilities) {
    return capabilities != Capability::none;
}

struct RunePrefix {
    std::uint16_t version{rune_version};
    RuneKind kind{RuneKind::tensor};
    Element element{Element::float64};
    std::uint32_t flags{0};
    std::uint16_t dimensions{0};
    std::uint32_t name_bytes{0};
    std::uint32_t shape_bytes{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t exchange_index{0};
    std::uint32_t rank{0};
    std::uint32_t peers{1};
    double physical_time{0.0};
    std::uint64_t session_id{0};
    std::uint64_t maximum_payload{0};
    std::uint64_t solver_time_index{0};
};

[[nodiscard]] std::array<std::byte, rune_prefix_size> encode_prefix(
    const RunePrefix& prefix);

[[nodiscard]] RunePrefix decode_prefix(
    std::span<const std::byte, rune_prefix_size> bytes);

[[nodiscard]] std::vector<std::byte> encode_shape(
    std::span<const std::uint64_t> shape);

[[nodiscard]] std::vector<std::uint64_t> decode_shape(
    std::span<const std::byte> bytes,
    std::uint16_t dimensions);

[[nodiscard]] std::vector<std::byte> encode_tensor(const TensorView& tensor);

[[nodiscard]] Tensor decode_tensor(std::span<const std::byte> frame);

}  // namespace foamnordic::fjord
