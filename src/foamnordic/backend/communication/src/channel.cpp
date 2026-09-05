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

#include "foamnordic/fjord/channel.hpp"

#include "foamnordic/fjord/shm_ring.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>

namespace foamnordic::fjord {

std::unique_ptr<FjordChannel> FjordChannel::duplicate() const {
    throw std::logic_error("This Fjord channel cannot be duplicated.");
}

namespace {

struct ShmOwner {
    void* first;
    void* second;
    std::atomic<bool> closed{false};
    std::atomic<std::uint64_t> data_epoch[2]{};
    std::atomic<std::uint64_t> space_epoch[2]{};
    std::mutex timed_wait_mutex;
    std::condition_variable timed_wait;
    std::atomic<std::uint32_t> timed_waiters{0};

    explicit ShmOwner(std::size_t size)
        : first(::operator new(size, std::align_val_t{SharedSlotRing::alignment})),
          second(::operator new(size, std::align_val_t{SharedSlotRing::alignment})) {}
    ~ShmOwner() {
        ::operator delete(first, std::align_val_t{SharedSlotRing::alignment});
        ::operator delete(second, std::align_val_t{SharedSlotRing::alignment});
    }
};

class SharedMemoryChannel final : public FjordChannel {
public:
    SharedMemoryChannel(
        std::shared_ptr<ShmOwner> owner,
        SharedSlotRing outbound,
        SharedSlotRing inbound,
        std::size_t outbound_index,
        std::size_t inbound_index)
        : owner_(std::move(owner)),
          outbound_(outbound),
          inbound_(inbound),
          outbound_index_(outbound_index),
          inbound_index_(inbound_index) {}

    void write_all(std::span<const std::byte> bytes) override {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto count = std::min<std::size_t>(
                outbound_.maximum_payload(), bytes.size() - offset);
            while (!outbound_.try_push(bytes.subspan(offset, count))) {
                if (owner_->closed.load(std::memory_order_acquire)) {
                    throw std::runtime_error("SHM channel closed while writing.");
                }
                const auto epoch = owner_->space_epoch[outbound_index_].load(
                    std::memory_order_acquire);
                if (!outbound_.writable()) {
                    owner_->space_epoch[outbound_index_].wait(
                        epoch, std::memory_order_acquire);
                }
            }
            owner_->data_epoch[outbound_index_].fetch_add(1, std::memory_order_release);
            owner_->data_epoch[outbound_index_].notify_one();
            if (owner_->timed_waiters.load(std::memory_order_acquire) != 0) {
                std::lock_guard lock(owner_->timed_wait_mutex);
                owner_->timed_wait.notify_all();
            }
            offset += count;
        }
    }

    void read_all(std::span<std::byte> bytes) override {
        std::size_t written = 0;
        while (written < bytes.size()) {
            std::size_t received = 0;
            bool complete = false;
            while (!inbound_.try_read_into(
                bytes.subspan(written), pending_offset_, received, complete)) {
                if (owner_->closed.load(std::memory_order_acquire)) {
                    throw std::runtime_error("SHM channel closed while reading.");
                }
                const auto epoch = owner_->data_epoch[inbound_index_].load(
                    std::memory_order_acquire);
                if (!inbound_.readable()) {
                    owner_->data_epoch[inbound_index_].wait(
                        epoch, std::memory_order_acquire);
                }
            }
            if (complete) {
                pending_offset_ = 0;
                owner_->space_epoch[inbound_index_].fetch_add(1, std::memory_order_release);
                owner_->space_epoch[inbound_index_].notify_one();
            } else {
                pending_offset_ += received;
            }
            written += received;
        }
    }

    bool wait_readable(std::chrono::milliseconds timeout) override {
        if (inbound_.readable()) {
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::unique_lock lock(owner_->timed_wait_mutex);
        owner_->timed_waiters.fetch_add(1, std::memory_order_release);
        if (!owner_->timed_wait.wait_until(lock, deadline, [this] {
                return inbound_.readable()
                       || owner_->closed.load(std::memory_order_acquire);
            })) {
            owner_->timed_waiters.fetch_sub(1, std::memory_order_release);
            return false;
        }
        owner_->timed_waiters.fetch_sub(1, std::memory_order_release);
        return inbound_.readable();
    }

    void interrupt() noexcept override {
        owner_->closed.store(true, std::memory_order_release);
        for (std::size_t ring = 0; ring < 2; ++ring) {
            owner_->data_epoch[ring].fetch_add(1, std::memory_order_release);
            owner_->space_epoch[ring].fetch_add(1, std::memory_order_release);
            owner_->data_epoch[ring].notify_all();
            owner_->space_epoch[ring].notify_all();
        }
        {
            std::lock_guard lock(owner_->timed_wait_mutex);
            owner_->timed_wait.notify_all();
        }
    }

    void close() noexcept override { interrupt(); }

private:
    std::shared_ptr<ShmOwner> owner_;
    SharedSlotRing outbound_;
    SharedSlotRing inbound_;
    std::size_t outbound_index_;
    std::size_t inbound_index_;
    std::size_t pending_offset_{0};
};

}  // namespace
namespace {

[[noreturn]] void system_failure(const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

}  // namespace

SocketChannel::SocketChannel(int descriptor) : descriptor_(descriptor) {
    if (descriptor_ < 0) {
        throw std::invalid_argument("Socket descriptor must be valid.");
    }
#ifdef SO_NOSIGPIPE
    const int enabled = 1;
    if (::setsockopt(descriptor_, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) {
        close();
        system_failure("Cannot configure Fjord socket safety");
    }
#endif
}

SocketChannel::~SocketChannel() { close(); }

SocketChannel::SocketChannel(SocketChannel&& other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)) {}

SocketChannel& SocketChannel::operator=(SocketChannel&& other) noexcept {
    if (this != &other) {
        close();
        descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
}

void SocketChannel::write_all(std::span<const std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const auto written = ::send(
            descriptor_, bytes.data() + offset, bytes.size() - offset, flags);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            system_failure("Fjord socket write failed");
        }
        if (written == 0) {
            throw std::runtime_error("Fjord socket closed during write.");
        }
        offset += static_cast<std::size_t>(written);
    }
}

void SocketChannel::read_all(std::span<std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto received =
            ::recv(descriptor_, bytes.data() + offset, bytes.size() - offset, 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            system_failure("Fjord socket read failed");
        }
        if (received == 0) {
            throw std::runtime_error("Fjord socket closed during read.");
        }
        offset += static_cast<std::size_t>(received);
    }
}

bool SocketChannel::wait_readable(std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) {
        throw std::invalid_argument("Fjord readiness timeout must not be negative.");
    }
    pollfd descriptor{descriptor_, POLLIN, 0};
    const auto bounded = std::min<std::int64_t>(timeout.count(), INT_MAX);
    int result = -1;
    do {
        result = ::poll(&descriptor, 1, static_cast<int>(bounded));
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
        system_failure("Fjord readiness wait failed");
    }
    if (result == 0) {
        return false;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        throw std::runtime_error("Fjord channel closed while waiting for data.");
    }
    return (descriptor.revents & POLLIN) != 0;
}

void SocketChannel::close() noexcept {
    if (descriptor_ >= 0) {
        ::close(descriptor_);
        descriptor_ = -1;
    }
}

void SocketChannel::interrupt() noexcept {
    if (descriptor_ >= 0) {
        static_cast<void>(::shutdown(descriptor_, SHUT_RDWR));
    }
}

int SocketChannel::descriptor() const noexcept { return descriptor_; }

std::unique_ptr<FjordChannel> SocketChannel::duplicate() const {
    const auto copied = ::dup(descriptor_);
    if (copied < 0) {
        system_failure("Fjord socket duplication failed");
    }
    return std::make_unique<SocketChannel>(copied);
}

std::pair<std::unique_ptr<FjordChannel>, std::unique_ptr<FjordChannel>>
local_channel_pair() {
    int descriptors[2]{};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0) {
        system_failure("Fjord local socket pair creation failed");
    }
    return {
        std::make_unique<SocketChannel>(descriptors[0]),
        std::make_unique<SocketChannel>(descriptors[1]),
    };
}

std::pair<std::unique_ptr<FjordChannel>, std::unique_ptr<FjordChannel>>
shared_memory_channel_pair(std::uint64_t slots, std::uint64_t slot_payload) {
    const auto bytes = SharedSlotRing::memory_size(slots, slot_payload);
    auto owner = std::make_shared<ShmOwner>(bytes);
    auto first = SharedSlotRing::initialize(owner->first, bytes, slots, slot_payload);
    auto second = SharedSlotRing::initialize(owner->second, bytes, slots, slot_payload);
    return {
        std::make_unique<SharedMemoryChannel>(owner, first, second, 0, 1),
        std::make_unique<SharedMemoryChannel>(owner, second, first, 1, 0),
    };
}

}  // namespace foamnordic::fjord
