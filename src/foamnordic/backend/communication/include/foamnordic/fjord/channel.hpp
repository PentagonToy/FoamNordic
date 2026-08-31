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

#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace foamnordic::fjord {

class FjordChannel {
public:
    virtual ~FjordChannel() = default;

    virtual void write_all(std::span<const std::byte> bytes) = 0;
    virtual void read_all(std::span<std::byte> bytes) = 0;
    [[nodiscard]] virtual bool wait_readable(std::chrono::milliseconds timeout) = 0;
    virtual void interrupt() noexcept = 0;
    virtual void close() noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<FjordChannel> duplicate() const;
};

class SocketChannel final : public FjordChannel {
public:
    explicit SocketChannel(int descriptor);
    ~SocketChannel() override;

    SocketChannel(const SocketChannel&) = delete;
    SocketChannel& operator=(const SocketChannel&) = delete;
    SocketChannel(SocketChannel&& other) noexcept;
    SocketChannel& operator=(SocketChannel&& other) noexcept;

    void write_all(std::span<const std::byte> bytes) override;
    void read_all(std::span<std::byte> bytes) override;
    [[nodiscard]] bool wait_readable(std::chrono::milliseconds timeout) override;
    void interrupt() noexcept override;
    void close() noexcept override;
    [[nodiscard]] std::unique_ptr<FjordChannel> duplicate() const override;

    [[nodiscard]] int descriptor() const noexcept;

private:
    int descriptor_{-1};
};

[[nodiscard]] std::pair<std::unique_ptr<FjordChannel>, std::unique_ptr<FjordChannel>>
local_channel_pair();

[[nodiscard]] std::pair<std::unique_ptr<FjordChannel>, std::unique_ptr<FjordChannel>>
shared_memory_channel_pair(
    std::uint64_t slots = 16,
    std::uint64_t slot_payload = 1024 * 1024);

}  // namespace foamnordic::fjord
