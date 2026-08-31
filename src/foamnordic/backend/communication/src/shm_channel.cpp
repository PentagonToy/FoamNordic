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

#include "foamnordic/fjord/shm_channel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#include "foamnordic/fjord/shm_ring.hpp"

namespace foamnordic::fjord {
namespace {

constexpr std::uint64_t region_magic = 0x464e53484d443031ULL;

std::size_t align_up(std::size_t value) {
    return (value + SharedSlotRing::alignment - 1) & ~(SharedSlotRing::alignment - 1);
}

struct alignas(SharedSlotRing::alignment) RegionHeader {
    std::uint64_t magic;
    std::uint64_t total_bytes;
    std::uint64_t ring_bytes;
    std::atomic<std::uint32_t> ready;
    std::atomic<std::uint32_t> closed;
    std::atomic<std::uint32_t> data_waiter[2];
    std::atomic<std::uint32_t> space_waiter[2];
};

constexpr std::byte wake_token(bool data, std::size_t ring) {
    return static_cast<std::byte>(1U + static_cast<unsigned>(ring) * 2U
                                  + (data ? 0U : 1U));
}

#if defined(__linux__)
int futex_wait(
    std::atomic<std::uint32_t>& word,
    const std::optional<std::chrono::milliseconds>& timeout) {
    timespec duration{};
    timespec* duration_pointer = nullptr;
    if (timeout) {
        duration.tv_sec = timeout->count() / 1000;
        duration.tv_nsec = (timeout->count() % 1000) * 1'000'000;
        duration_pointer = &duration;
    }
    return static_cast<int>(::syscall(
        SYS_futex,
        reinterpret_cast<std::uint32_t*>(&word),
        FUTEX_WAIT,
        1,
        duration_pointer,
        nullptr,
        0));
}

void futex_wake(std::atomic<std::uint32_t>& word) noexcept {
    static_cast<void>(::syscall(
        SYS_futex,
        reinterpret_cast<std::uint32_t*>(&word),
        FUTEX_WAKE,
        1,
        nullptr,
        nullptr,
        0));
}
#endif

void validate_name(const std::string& name) {
    if (name.size() < 2 || name.front() != '/' || name.find('/', 1) != std::string::npos) {
        throw std::invalid_argument(
            "POSIX SHM name must begin with one slash and contain no other slash.");
    }
}

class Mapping {
public:
    Mapping(std::string name, int descriptor, void* address, std::size_t bytes, bool owner)
        : name_(std::move(name)), descriptor_(descriptor), address_(address), bytes_(bytes), owner_(owner) {}
    ~Mapping() {
        if (address_ != MAP_FAILED) {
            ::munmap(address_, bytes_);
        }
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
        if (owner_) {
            ::shm_unlink(name_.c_str());
        }
    }
    [[nodiscard]] std::byte* bytes() const noexcept { return static_cast<std::byte*>(address_); }
    [[nodiscard]] RegionHeader* header() const noexcept { return static_cast<RegionHeader*>(address_); }

private:
    std::string name_;
    int descriptor_;
    void* address_;
    std::size_t bytes_;
    bool owner_;
};

class NamedShmChannel final : public FjordChannel {
public:
    NamedShmChannel(
        std::shared_ptr<Mapping> mapping,
        SharedSlotRing outbound,
        SharedSlotRing inbound,
        std::size_t outbound_index,
        std::size_t inbound_index,
        std::unique_ptr<FjordChannel> wake_channel)
        : mapping_(std::move(mapping)),
          outbound_(outbound),
          inbound_(inbound),
          outbound_index_(outbound_index),
          inbound_index_(inbound_index),
          wake_channel_(std::move(wake_channel)) {
        if (!wake_channel_) {
            throw std::invalid_argument(
                "POSIX SHM requires a blocking wake channel; use a connected UDS channel.");
        }
    }

    void write_all(std::span<const std::byte> bytes) override {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto count = std::min<std::size_t>(outbound_.maximum_payload(), bytes.size() - offset);
            while (!outbound_.try_push(bytes.subspan(offset, count))) {
                require_open("writing");
                wait_for_space();
            }
            notify_data(outbound_index_);
            offset += count;
        }
    }

    void read_all(std::span<std::byte> bytes) override {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            if (pending_offset_ == pending_.size()) {
                pending_.clear();
                pending_offset_ = 0;
                while (!inbound_.try_pop(pending_)) {
                    require_open("reading");
                    wait_for_data(std::nullopt);
                }
                notify_space(inbound_index_);
            }
            const auto count = std::min(bytes.size() - offset, pending_.size() - pending_offset_);
            std::memcpy(bytes.data() + offset, pending_.data() + pending_offset_, count);
            offset += count;
            pending_offset_ += count;
        }
    }

    bool wait_readable(std::chrono::milliseconds timeout) override {
        if (pending_offset_ < pending_.size()) {
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            if (inbound_.try_pop(pending_)) {
                pending_offset_ = 0;
                notify_space(inbound_index_);
                return true;
            }
            require_open("waiting");
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return false;
            }
            if (!wait_for_data(
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now))) {
                return false;
            }
        }
    }

    void interrupt() noexcept override {
        mapping_->header()->closed.store(1, std::memory_order_release);
#if defined(__linux__)
        for (std::size_t ring = 0; ring < 2; ++ring) {
            mapping_->header()->data_waiter[ring].store(0, std::memory_order_release);
            mapping_->header()->space_waiter[ring].store(0, std::memory_order_release);
            futex_wake(mapping_->header()->data_waiter[ring]);
            futex_wake(mapping_->header()->space_waiter[ring]);
        }
#endif
        wake_channel_->interrupt();
    }

    void close() noexcept override {
        interrupt();
        wake_channel_->close();
    }

private:
    void require_open(const char* operation) const {
        if (mapping_->header()->closed.load(std::memory_order_acquire) != 0) {
            throw std::runtime_error(std::string("POSIX SHM channel closed while ") + operation + '.');
        }
    }

    void notify(std::atomic<std::uint32_t>& waiter, std::byte token) {
        if (waiter.exchange(0, std::memory_order_acq_rel) != 0) {
#if defined(__linux__)
            (void)token;
            futex_wake(waiter);
#else
            wake_channel_->write_all(std::span<const std::byte>(&token, 1));
#endif
        }
    }

    void notify_data(std::size_t ring) {
        notify(mapping_->header()->data_waiter[ring], wake_token(true, ring));
    }

    void notify_space(std::size_t ring) {
        notify(mapping_->header()->space_waiter[ring], wake_token(false, ring));
    }

    bool receive_wake(
        std::atomic<std::uint32_t>& waiter,
        std::byte expected,
        std::optional<std::chrono::milliseconds> timeout) {
#if defined(__linux__)
        (void)expected;
        const auto result = futex_wait(waiter, timeout);
        if (result == 0 || errno == EAGAIN || errno == EINTR) {
            return true;
        }
        if (errno == ETIMEDOUT) {
            return false;
        }
        throw std::system_error(errno, std::generic_category(), "SHM futex wait failed");
#else
        const auto index = static_cast<unsigned>(expected);
        if (pending_wakes_[index] != 0) {
            --pending_wakes_[index];
            return true;
        }
        if (timeout && !wake_channel_->wait_readable(*timeout)) {
            return false;
        }
        while (true) {
            std::byte token{};
            wake_channel_->read_all(std::span<std::byte>(&token, 1));
            const auto received = static_cast<unsigned>(token);
            if (received == index) {
                return true;
            }
            if (received < pending_wakes_.size()) {
                ++pending_wakes_[received];
            }
            if (timeout && !wake_channel_->wait_readable(*timeout)) {
                return false;
            }
        }
#endif
    }

    bool wait_for_data(std::optional<std::chrono::milliseconds> timeout) {
        auto& waiter = mapping_->header()->data_waiter[inbound_index_];
        waiter.store(1, std::memory_order_release);
        if (inbound_.readable()) {
            waiter.store(0, std::memory_order_release);
            return true;
        }
        return receive_wake(waiter, wake_token(true, inbound_index_), timeout);
    }

    void wait_for_space() {
        auto& waiter = mapping_->header()->space_waiter[outbound_index_];
        waiter.store(1, std::memory_order_release);
        if (outbound_.writable()) {
            waiter.store(0, std::memory_order_release);
            return;
        }
        static_cast<void>(receive_wake(
            waiter, wake_token(false, outbound_index_), std::nullopt));
    }

    std::shared_ptr<Mapping> mapping_;
    SharedSlotRing outbound_;
    SharedSlotRing inbound_;
    std::size_t outbound_index_;
    std::size_t inbound_index_;
    std::unique_ptr<FjordChannel> wake_channel_;
    std::array<std::uint32_t, 5> pending_wakes_{};
    std::vector<std::byte> pending_;
    std::size_t pending_offset_{0};
};

std::unique_ptr<FjordChannel> make_channel(
    std::shared_ptr<Mapping> mapping,
    SharedMemoryRole role,
    std::unique_ptr<FjordChannel> wake_channel) {
    const auto header_bytes = align_up(sizeof(RegionHeader));
    auto* first_address = mapping->bytes() + header_bytes;
    auto* second_address = first_address + mapping->header()->ring_bytes;
    auto first = SharedSlotRing::attach(first_address, mapping->header()->ring_bytes);
    auto second = SharedSlotRing::attach(second_address, mapping->header()->ring_bytes);
    return role == SharedMemoryRole::initiator
               ? std::make_unique<NamedShmChannel>(
                     mapping, first, second, 0, 1, std::move(wake_channel))
               : std::make_unique<NamedShmChannel>(
                     mapping, second, first, 1, 0, std::move(wake_channel));
}

}  // namespace

std::unique_ptr<FjordChannel> create_shared_memory_channel(
    const std::string& name,
    std::unique_ptr<FjordChannel> wake_channel,
    SharedMemoryRole role,
    std::uint64_t slots,
    std::uint64_t slot_payload) {
    validate_name(name);
    const auto ring_bytes = SharedSlotRing::memory_size(slots, slot_payload);
    const auto total_bytes = align_up(sizeof(RegionHeader)) + 2 * ring_bytes;
    const auto descriptor = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (descriptor < 0) {
        throw std::system_error(errno, std::generic_category(), "POSIX SHM creation failed");
    }
    if (::ftruncate(descriptor, static_cast<off_t>(total_bytes)) != 0) {
        const auto error = errno;
        ::close(descriptor);
        ::shm_unlink(name.c_str());
        throw std::system_error(error, std::generic_category(), "POSIX SHM sizing failed");
    }
    void* address = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    if (address == MAP_FAILED) {
        const auto error = errno;
        ::close(descriptor);
        ::shm_unlink(name.c_str());
        throw std::system_error(error, std::generic_category(), "POSIX SHM mapping failed");
    }
    auto mapping = std::make_shared<Mapping>(name, descriptor, address, total_bytes, true);
    auto* header = std::construct_at(mapping->header());
    header->magic = region_magic;
    header->total_bytes = total_bytes;
    header->ring_bytes = ring_bytes;
    header->ready.store(0, std::memory_order_relaxed);
    header->closed.store(0, std::memory_order_relaxed);
    for (std::size_t ring = 0; ring < 2; ++ring) {
        header->data_waiter[ring].store(0, std::memory_order_relaxed);
        header->space_waiter[ring].store(0, std::memory_order_relaxed);
    }
    auto* first = mapping->bytes() + align_up(sizeof(RegionHeader));
    static_cast<void>(SharedSlotRing::initialize(first, ring_bytes, slots, slot_payload));
    static_cast<void>(SharedSlotRing::initialize(first + ring_bytes, ring_bytes, slots, slot_payload));
    header->ready.store(1, std::memory_order_release);
    return make_channel(std::move(mapping), role, std::move(wake_channel));
}

std::unique_ptr<FjordChannel> connect_shared_memory_channel(
    const std::string& name,
    std::unique_ptr<FjordChannel> wake_channel,
    SharedMemoryRole role) {
    validate_name(name);
    const auto descriptor = ::shm_open(name.c_str(), O_RDWR, 0600);
    if (descriptor < 0) {
        throw std::system_error(errno, std::generic_category(), "POSIX SHM connection failed");
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(RegionHeader))) {
        const auto error = errno == 0 ? EINVAL : errno;
        ::close(descriptor);
        throw std::system_error(error, std::generic_category(), "POSIX SHM metadata failed");
    }
    const auto bytes = static_cast<std::size_t>(status.st_size);
    void* address = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    if (address == MAP_FAILED) {
        const auto error = errno;
        ::close(descriptor);
        throw std::system_error(error, std::generic_category(), "POSIX SHM mapping failed");
    }
    auto mapping = std::make_shared<Mapping>(name, descriptor, address, bytes, false);
    const auto* header = mapping->header();
    if (header->ready.load(std::memory_order_acquire) != 1 || header->magic != region_magic
        || header->total_bytes > bytes
        || align_up(sizeof(RegionHeader)) + 2 * header->ring_bytes
               != header->total_bytes) {
        throw std::runtime_error(
            "POSIX SHM region is incomplete or incompatible: ready="
            + std::to_string(header->ready.load(std::memory_order_relaxed))
            + ", bytes=" + std::to_string(bytes)
            + ", declared=" + std::to_string(header->total_bytes)
            + ", ring=" + std::to_string(header->ring_bytes) + '.');
    }
    return make_channel(std::move(mapping), role, std::move(wake_channel));
}

}  // namespace foamnordic::fjord
