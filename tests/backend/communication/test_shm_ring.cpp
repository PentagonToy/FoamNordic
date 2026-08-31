#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "foamnordic/fjord/harbor.hpp"
#include "foamnordic/fjord/shm_channel.hpp"
#include "foamnordic/fjord/shm_ring.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class AlignedMemory {
public:
    explicit AlignedMemory(std::size_t bytes)
        : data_(::operator new(
              bytes,
              std::align_val_t{foamnordic::fjord::SharedSlotRing::alignment})),
          bytes_(bytes) {}
    ~AlignedMemory() {
        ::operator delete(
            data_, std::align_val_t{foamnordic::fjord::SharedSlotRing::alignment});
    }
    AlignedMemory(const AlignedMemory&) = delete;
    AlignedMemory& operator=(const AlignedMemory&) = delete;
    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
    void* data_;
    std::size_t bytes_;
};

std::vector<std::byte> message(std::uint64_t sequence, std::size_t padding) {
    std::vector<std::byte> bytes(sizeof(sequence) + padding);
    std::memcpy(bytes.data(), &sequence, sizeof(sequence));
    for (std::size_t index = sizeof(sequence); index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((sequence + index) & 0xffU);
    }
    return bytes;
}

void test_capacity_and_backpressure() {
    const auto bytes = foamnordic::fjord::SharedSlotRing::memory_size(2, 64);
    AlignedMemory memory(bytes);
    auto writer = foamnordic::fjord::SharedSlotRing::initialize(
        memory.data(), memory.size(), 2, 64);
    auto reader = foamnordic::fjord::SharedSlotRing::attach(memory.data(), memory.size());
    const auto first = message(1, 8);
    const auto second = message(2, 8);
    require(writer.try_push(first), "SHM ring rejected its first slot.");
    require(writer.try_push(second), "SHM ring rejected its second slot.");
    require(!writer.try_push(first), "SHM ring overwrote an unread slot.");
    std::vector<std::byte> received;
    require(reader.try_pop(received) && received == first, "SHM ring first slot is incorrect.");
    require(writer.try_push(first), "SHM ring did not reuse a released slot.");
}

void test_concurrent_wraparound() {
    constexpr std::uint64_t exchanges = 100'000;
    const auto bytes = foamnordic::fjord::SharedSlotRing::memory_size(8, 256);
    AlignedMemory memory(bytes);
    auto writer = foamnordic::fjord::SharedSlotRing::initialize(
        memory.data(), memory.size(), 8, 256);
    auto reader = foamnordic::fjord::SharedSlotRing::attach(memory.data(), memory.size());
    std::atomic<bool> failed{false};

    std::thread producer([&] {
        for (std::uint64_t sequence = 0; sequence < exchanges; ++sequence) {
            const auto outbound = message(sequence, sequence % 193);
            while (!writer.try_push(outbound)) {
                std::this_thread::yield();
            }
        }
    });
    std::thread consumer([&] {
        std::vector<std::byte> inbound;
        for (std::uint64_t sequence = 0; sequence < exchanges; ++sequence) {
            while (!reader.try_pop(inbound)) {
                std::this_thread::yield();
            }
            if (inbound != message(sequence, sequence % 193)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    });
    producer.join();
    consumer.join();
    require(!failed.load(std::memory_order_relaxed), "SHM ring exposed corrupt or partial data.");
}

void test_named_cross_process_harbor() {
    const auto name = std::string("/foamnordic-") + std::to_string(::getpid());
    auto wake_channels = foamnordic::fjord::local_channel_pair();
    auto parent_channel = foamnordic::fjord::create_shared_memory_channel(
        name, std::move(wake_channels.first));
    const auto child = ::fork();
    require(child >= 0, "Could not fork POSIX SHM test peer.");
    if (child == 0) {
        try {
            foamnordic::fjord::Harbor worker(
                foamnordic::fjord::connect_shared_memory_channel(
                    name, std::move(wake_channels.second)));
            static_cast<void>(worker.accept_session({
                1,
                foamnordic::fjord::Capability::shm,
                0,
                1,
                4096,
            }));
            auto tensor = worker.receive();
            worker.send(tensor.view());
            _exit(0);
        } catch (const std::exception& error) {
            ::write(STDERR_FILENO, error.what(), std::strlen(error.what()));
            ::write(STDERR_FILENO, "\n", 1);
            _exit(2);
        }
    }

    wake_channels.second.reset();

    foamnordic::fjord::Harbor parent(std::move(parent_channel));
    static_cast<void>(parent.connect_session({
        81,
        foamnordic::fjord::Capability::shm,
        0,
        1,
        4096,
    }));
    const std::vector<double> values{0.25, 0.5, 0.75, 1.0};
    parent.send({
        "c_tilde",
        foamnordic::fjord::Element::float64,
        {4},
        foamnordic::fjord::as_bytes(std::span(values)),
        12,
        0.3,
    });
    const auto echoed = parent.receive();
    require(echoed.bytes.size() == values.size() * sizeof(double), "SHM Harbor payload size failed.");
    require(
        std::memcmp(echoed.bytes.data(), values.data(), echoed.bytes.size()) == 0,
        "SHM Harbor cross-process payload failed.");
    int status = 0;
    require(::waitpid(child, &status, 0) == child, "Could not wait for SHM test peer.");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "SHM test peer failed.");
}

void test_blocking_uds_to_shm_upgrade() {
    const auto name = std::string("/foamnordic-upgrade-") + std::to_string(::getpid());
    auto channels = foamnordic::fjord::local_channel_pair();
    foamnordic::fjord::Harbor client(std::move(channels.first));
    foamnordic::fjord::Harbor server(std::move(channels.second));

    std::thread worker([&server, &name] {
        static_cast<void>(server.accept_session({
            91,
            foamnordic::fjord::Capability::uds | foamnordic::fjord::Capability::shm,
            0,
            1,
            4096,
        }));
        server.offer_shared_memory(name);
        auto tensor = server.receive();
        const auto view = tensor.view();
        server.publish(tensor.time_index, std::span(&view, 1));
    });

    const auto selected = client.connect_session({
        91,
        foamnordic::fjord::Capability::uds | foamnordic::fjord::Capability::shm,
        0,
        1,
        4096,
    });
    require(
        foamnordic::fjord::any(selected.capabilities & foamnordic::fjord::Capability::shm),
        "SHM capability was not negotiated.");
    client.accept_shared_memory();

    const std::array<double, 2> values{0.25, 0.75};
    const foamnordic::fjord::TensorView input{
        "c_tilde",
        foamnordic::fjord::Element::float64,
        {2},
        foamnordic::fjord::as_bytes(std::span(values)),
        5,
        0.2,
    };
    client.send(input);
    const auto output = client.receive_message();
    const auto commit = client.receive_message();
    require(
        output.kind == foamnordic::fjord::RuneKind::tensor
            && output.tensor && output.tensor->bytes.size() == sizeof(values),
        "Upgraded SHM channel did not return its tensor.");
    require(
        commit.kind == foamnordic::fjord::RuneKind::complete
            && commit.exchange_index == 5 && commit.tensor_count == 1,
        "Upgraded SHM channel did not preserve atomic publication.");
    worker.join();
}

}  // namespace

int main() {
    test_capacity_and_backpressure();
    test_concurrent_wraparound();
    test_named_cross_process_harbor();
    test_blocking_uds_to_shm_upgrade();
}
