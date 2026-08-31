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

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace foamnordic::fjord {

class SharedSlotRing {
public:
    static constexpr std::size_t alignment = 64;

    [[nodiscard]] static std::size_t memory_size(
        std::uint64_t slots,
        std::uint64_t payload_bytes);
    [[nodiscard]] static SharedSlotRing initialize(
        void* memory,
        std::size_t bytes,
        std::uint64_t slots,
        std::uint64_t payload_bytes);
    [[nodiscard]] static SharedSlotRing attach(void* memory, std::size_t bytes);

    [[nodiscard]] bool try_push(std::span<const std::byte> message);
    [[nodiscard]] bool try_pop(std::vector<std::byte>& message);
    [[nodiscard]] bool readable() const noexcept;
    [[nodiscard]] bool writable() const noexcept;

    [[nodiscard]] std::uint64_t capacity() const noexcept;
    [[nodiscard]] std::uint64_t maximum_payload() const noexcept;

private:
    struct Header;
    struct Slot;

    SharedSlotRing(void* memory, std::size_t bytes);
    [[nodiscard]] Slot* slot(std::uint64_t position) const noexcept;

    std::byte* memory_;
    Header* header_;
};

}  // namespace foamnordic::fjord
