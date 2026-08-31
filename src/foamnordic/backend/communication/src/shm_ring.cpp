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

#include "foamnordic/fjord/shm_ring.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

namespace foamnordic::fjord {
namespace {

constexpr std::uint64_t ring_magic = 0x464e53484d523031ULL;

std::size_t align_up(std::size_t value) {
    return (value + SharedSlotRing::alignment - 1) & ~(SharedSlotRing::alignment - 1);
}

}  // namespace

struct alignas(SharedSlotRing::alignment) SharedSlotRing::Header {
    std::uint64_t magic;
    std::uint64_t slots;
    std::uint64_t payload_bytes;
    std::uint64_t slot_stride;
    std::atomic<std::uint64_t> write_position;
    std::atomic<std::uint64_t> read_position;
};

struct alignas(SharedSlotRing::alignment) SharedSlotRing::Slot {
    std::atomic<std::uint64_t> sequence;
    std::uint64_t size;
};

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

std::size_t SharedSlotRing::memory_size(
    std::uint64_t slots,
    std::uint64_t payload_bytes) {
    if (slots < 2 || payload_bytes == 0) {
        throw std::invalid_argument("SHM ring requires two slots and a positive payload.");
    }
    const auto stride = align_up(sizeof(Slot) + payload_bytes);
    if (slots > (std::numeric_limits<std::size_t>::max() - align_up(sizeof(Header))) / stride) {
        throw std::overflow_error("SHM ring allocation size overflowed.");
    }
    return align_up(sizeof(Header)) + static_cast<std::size_t>(slots) * stride;
}

SharedSlotRing::SharedSlotRing(void* memory, std::size_t bytes)
    : memory_(static_cast<std::byte*>(memory)),
      header_(static_cast<Header*>(memory)) {
    (void)bytes;
}

SharedSlotRing SharedSlotRing::initialize(
    void* memory,
    std::size_t bytes,
    std::uint64_t slots,
    std::uint64_t payload_bytes) {
    if (memory == nullptr || reinterpret_cast<std::uintptr_t>(memory) % alignment != 0
        || bytes < memory_size(slots, payload_bytes)) {
        throw std::invalid_argument("SHM ring memory is missing, unaligned, or too small.");
    }
    auto* header = std::construct_at(static_cast<Header*>(memory));
    header->magic = ring_magic;
    header->slots = slots;
    header->payload_bytes = payload_bytes;
    header->slot_stride = align_up(sizeof(Slot) + payload_bytes);
    header->write_position.store(0, std::memory_order_relaxed);
    header->read_position.store(0, std::memory_order_relaxed);
    SharedSlotRing ring(memory, bytes);
    for (std::uint64_t index = 0; index < slots; ++index) {
        auto* current = std::construct_at(ring.slot(index));
        current->sequence.store(index, std::memory_order_relaxed);
        current->size = 0;
    }
    std::atomic_thread_fence(std::memory_order_release);
    return ring;
}

SharedSlotRing SharedSlotRing::attach(void* memory, std::size_t bytes) {
    if (memory == nullptr || reinterpret_cast<std::uintptr_t>(memory) % alignment != 0
        || bytes < sizeof(Header)) {
        throw std::invalid_argument("SHM ring attachment memory is invalid.");
    }
    SharedSlotRing ring(memory, bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (ring.header_->magic != ring_magic
        || bytes < memory_size(ring.header_->slots, ring.header_->payload_bytes)) {
        throw std::runtime_error("SHM ring header is invalid or incomplete.");
    }
    return ring;
}

SharedSlotRing::Slot* SharedSlotRing::slot(std::uint64_t position) const noexcept {
    const auto index = position % header_->slots;
    return reinterpret_cast<Slot*>(
        memory_ + align_up(sizeof(Header)) + index * header_->slot_stride);
}

bool SharedSlotRing::try_push(std::span<const std::byte> message) {
    if (message.size() > header_->payload_bytes) {
        throw std::length_error("SHM message exceeds its negotiated slot payload.");
    }
    const auto position = header_->write_position.load(std::memory_order_relaxed);
    auto* current = slot(position);
    if (current->sequence.load(std::memory_order_acquire) != position) {
        return false;
    }
    current->size = message.size();
    std::memcpy(reinterpret_cast<std::byte*>(current) + sizeof(Slot), message.data(), message.size());
    current->sequence.store(position + 1, std::memory_order_release);
    header_->write_position.store(position + 1, std::memory_order_relaxed);
    return true;
}

bool SharedSlotRing::try_pop(std::vector<std::byte>& message) {
    const auto position = header_->read_position.load(std::memory_order_relaxed);
    auto* current = slot(position);
    if (current->sequence.load(std::memory_order_acquire) != position + 1) {
        return false;
    }
    if (current->size > header_->payload_bytes) {
        throw std::runtime_error("SHM slot contains an invalid message size.");
    }
    message.resize(current->size);
    std::memcpy(message.data(), reinterpret_cast<std::byte*>(current) + sizeof(Slot), current->size);
    current->sequence.store(position + header_->slots, std::memory_order_release);
    header_->read_position.store(position + 1, std::memory_order_relaxed);
    return true;
}

bool SharedSlotRing::readable() const noexcept {
    const auto position = header_->read_position.load(std::memory_order_relaxed);
    return slot(position)->sequence.load(std::memory_order_acquire) == position + 1;
}

bool SharedSlotRing::writable() const noexcept {
    const auto position = header_->write_position.load(std::memory_order_relaxed);
    return slot(position)->sequence.load(std::memory_order_acquire) == position;
}

std::uint64_t SharedSlotRing::capacity() const noexcept { return header_->slots; }
std::uint64_t SharedSlotRing::maximum_payload() const noexcept {
    return header_->payload_bytes;
}

}  // namespace foamnordic::fjord
