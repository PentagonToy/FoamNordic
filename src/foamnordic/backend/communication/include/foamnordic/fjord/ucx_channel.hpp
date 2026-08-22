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

#ifndef FOAMNORDIC_HAVE_UCX
#error "FoamNordic UCX support was not enabled for this build."
#endif

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>

#include "foamnordic/fjord/channel.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace foamnordic::fjord {

class UcxChannel final : public FjordChannel {
public:
    ~UcxChannel() override;

    UcxChannel(const UcxChannel&) = delete;
    UcxChannel& operator=(const UcxChannel&) = delete;
    UcxChannel(UcxChannel&&) = delete;
    UcxChannel& operator=(UcxChannel&&) = delete;

    void write_all(std::span<const std::byte> bytes) override;
    void read_all(std::span<std::byte> bytes) override;
    [[nodiscard]] bool wait_readable(std::chrono::milliseconds timeout) override;
    void interrupt() noexcept override;
    void close() noexcept override;

private:
    struct Impl;
    explicit UcxChannel(std::unique_ptr<Impl> implementation);

    std::unique_ptr<Impl> implementation_;

    friend class UcxListener;
    friend std::unique_ptr<FjordChannel> connect_ucx(const FjordAddress& address);
};

class UcxListener {
public:
    [[nodiscard]] static UcxListener network(
        const std::string& host,
        std::uint16_t port);

    ~UcxListener();
    UcxListener(const UcxListener&) = delete;
    UcxListener& operator=(const UcxListener&) = delete;
    UcxListener(UcxListener&&) noexcept;
    UcxListener& operator=(UcxListener&&) noexcept;

    [[nodiscard]] std::unique_ptr<FjordChannel> accept(
        std::chrono::milliseconds timeout = std::chrono::seconds(30));
    [[nodiscard]] FjordAddress address() const;
    void close() noexcept;

private:
    struct Impl;
    explicit UcxListener(std::unique_ptr<Impl> implementation);

    std::unique_ptr<Impl> implementation_;
};

[[nodiscard]] std::unique_ptr<FjordChannel> connect_ucx(
    const FjordAddress& address);

}  // namespace foamnordic::fjord
