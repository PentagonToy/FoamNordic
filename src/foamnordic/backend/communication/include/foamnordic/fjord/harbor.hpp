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
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "foamnordic/fjord/channel.hpp"
#include "foamnordic/fjord/rune.hpp"
#include "foamnordic/fjord/tensor.hpp"

namespace foamnordic::fjord {

#ifdef FOAMNORDIC_HAVE_UCX
class UcxListener;
#endif

struct SessionHello {
    std::uint64_t session_id{0};
    Capability capabilities{Capability::none};
    std::uint32_t rank{0};
    std::uint32_t peers{1};
    std::uint64_t maximum_payload{16ULL * 1024ULL * 1024ULL * 1024ULL};

    void validate() const;
};

enum class HandshakeMode {
    blocking,
    timed,
    disabled,
};

struct HarborOptions {
    HandshakeMode handshake{HandshakeMode::blocking};
    std::chrono::milliseconds handshake_timeout{30'000};

    void validate() const;
};

struct HarborMessage {
    RuneKind kind{RuneKind::error};
    std::optional<Tensor> tensor;
    std::uint64_t exchange_index{0};
    std::uint32_t tensor_count{0};
};

class Harbor {
public:
    explicit Harbor(
        std::unique_ptr<FjordChannel> channel,
        HarborOptions options = {});

    Harbor(const Harbor&) = delete;
    Harbor& operator=(const Harbor&) = delete;
    Harbor(Harbor&&) noexcept = default;
    Harbor& operator=(Harbor&&) noexcept = default;

    void send(const TensorView& tensor);
    void publish(std::uint64_t exchange_index, std::span<const TensorView> tensors);
    [[nodiscard]] HarborMessage receive_message();
    [[nodiscard]] Tensor receive();
    [[nodiscard]] SessionHello connect_session(const SessionHello& offer);
    [[nodiscard]] SessionHello accept_session(const SessionHello& support);
    void complete(std::uint64_t exchange_index, std::uint32_t tensor_count);
    void fail_exchange(std::uint64_t exchange_index);
    void shutdown();
    void offer_shared_memory(const std::string& name);
    void accept_shared_memory();
#ifdef FOAMNORDIC_HAVE_UCX
    void offer_ucx(UcxListener& listener);
    void accept_ucx();
#endif
    [[nodiscard]] RuneKind receive_control(std::uint64_t* exchange_index = nullptr);
    [[nodiscard]] std::uint32_t rank() const noexcept;
    void interrupt() noexcept;
    void close() noexcept;

private:
    void send_control(
        RuneKind kind,
        const SessionHello& session,
        std::uint64_t exchange_index = 0,
        std::uint32_t tensor_count = 0);
    [[nodiscard]] RunePrefix receive_prefix();
    void wait_for_handshake();

    std::unique_ptr<FjordChannel> channel_;
    HarborOptions options_;
    SessionHello session_{};
};

}  // namespace foamnordic::fjord
