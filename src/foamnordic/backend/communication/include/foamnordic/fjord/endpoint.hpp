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

#include <cstdint>
#include <memory>
#include <string>

#include "foamnordic/fjord/channel.hpp"

namespace foamnordic::fjord {

enum class FjordKind {
    unix_socket,
    tcp,
};

struct FjordAddress {
    FjordKind kind{FjordKind::unix_socket};
    std::string location;
    std::uint16_t port{0};

    [[nodiscard]] static FjordAddress local(std::string path);
    [[nodiscard]] static FjordAddress network(std::string host, std::uint16_t port);
    [[nodiscard]] static FjordAddress parse(const std::string& text);
    [[nodiscard]] std::string text() const;
    void validate() const;
};

class FjordListener {
public:
    [[nodiscard]] static FjordListener local(const std::string& path, int backlog = 16);
    [[nodiscard]] static FjordListener network(
        const std::string& host,
        std::uint16_t port,
        int backlog = 16);

    ~FjordListener();
    FjordListener(const FjordListener&) = delete;
    FjordListener& operator=(const FjordListener&) = delete;
    FjordListener(FjordListener&& other) noexcept;
    FjordListener& operator=(FjordListener&& other) noexcept;

    [[nodiscard]] std::unique_ptr<FjordChannel> accept();
    [[nodiscard]] FjordAddress address() const;
    void close() noexcept;

private:
    FjordListener(int descriptor, FjordAddress address, bool owns_path);

    int descriptor_{-1};
    FjordAddress address_;
    bool owns_path_{false};
};

[[nodiscard]] std::unique_ptr<FjordChannel> connect(const FjordAddress& address);

}  // namespace foamnordic::fjord
