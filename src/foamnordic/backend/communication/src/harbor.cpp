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

#include "foamnordic/fjord/harbor.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "foamnordic/fjord/rune.hpp"
#include "foamnordic/fjord/shm_channel.hpp"
#ifdef FOAMNORDIC_HAVE_UCX
#include "foamnordic/fjord/ucx_channel.hpp"
#endif

namespace foamnordic::fjord {

void SessionHello::validate() const {
    if (session_id == 0) {
        throw std::invalid_argument("A FoamNordic session ID must be non-zero.");
    }
    if (!any(capabilities)) {
        throw std::invalid_argument("A FoamNordic session requires at least one capability.");
    }
    if (peers == 0 || rank >= peers) {
        throw std::invalid_argument("FoamNordic session rank metadata is invalid.");
    }
    if (maximum_payload == 0) {
        throw std::invalid_argument("FoamNordic maximum payload must be positive.");
    }
}

void HarborOptions::validate() const {
    if (handshake == HandshakeMode::timed && handshake_timeout.count() <= 0) {
        throw std::invalid_argument("A timed FoamNordic handshake requires a positive timeout.");
    }
}

Harbor::Harbor(std::unique_ptr<FjordChannel> channel, HarborOptions options)
    : channel_(std::move(channel)), options_(options) {
    if (!channel_) {
        throw std::invalid_argument("Harbor requires a Fjord channel.");
    }
    options_.validate();
}

void Harbor::send(const TensorView& tensor) {
    tensor.validate();
    if (tensor.shape.size() > 16 || tensor.name.size() > 4096) {
        throw std::invalid_argument("Tensor metadata exceeds the Rune safety limit.");
    }
    if (tensor.bytes.size() > session_.maximum_payload) {
        throw std::invalid_argument("Tensor payload exceeds the negotiated Harbor limit.");
    }
    const auto shape = encode_shape(tensor.shape);
    const RunePrefix prefix{
        .kind = RuneKind::tensor,
        .element = tensor.element,
        .dimensions = static_cast<std::uint16_t>(tensor.shape.size()),
        .name_bytes = static_cast<std::uint32_t>(tensor.name.size()),
        .shape_bytes = static_cast<std::uint32_t>(shape.size()),
        .payload_bytes = static_cast<std::uint64_t>(tensor.bytes.size()),
        .exchange_index = tensor.time_index,
        .rank = session_.rank,
        .peers = session_.peers,
        .physical_time = tensor.physical_time,
        .session_id = session_.session_id,
        .solver_time_index = tensor.solver_time_index,
    };
    const auto header = encode_prefix(prefix);
    channel_->write_all(header);
    channel_->write_all(std::as_bytes(std::span(tensor.name)));
    channel_->write_all(shape);
    channel_->write_all(tensor.bytes);
}

void Harbor::publish(
    std::uint64_t exchange_index,
    std::span<const TensorView> tensors) {
    if (tensors.empty()) {
        throw std::invalid_argument("An atomic exchange must contain at least one tensor.");
    }
    if (tensors.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("An atomic exchange contains too many tensors.");
    }
    for (const auto& tensor : tensors) {
        if (tensor.time_index != exchange_index) {
            throw std::invalid_argument(
                "Every tensor in an atomic exchange must share its exchange index.");
        }
        if (tensor.solver_time_index != tensors.front().solver_time_index
            || tensor.physical_time != tensors.front().physical_time) {
            throw std::invalid_argument(
                "Every tensor in an atomic exchange must share solver-time metadata.");
        }
        send(tensor);
    }
    complete(exchange_index, static_cast<std::uint32_t>(tensors.size()));
}

HarborMessage Harbor::receive_message() {
    const auto prefix = receive_prefix();
    if (session_.session_id != 0 && prefix.session_id != session_.session_id) {
        throw std::runtime_error("Harbor received data for a different session.");
    }
    if (prefix.kind != RuneKind::tensor) {
        if (prefix.kind != RuneKind::complete && prefix.kind != RuneKind::shutdown
            && prefix.kind != RuneKind::error) {
            throw std::runtime_error("Harbor received an unexpected Rune message.");
        }
        return {prefix.kind, std::nullopt, prefix.exchange_index, prefix.flags};
    }

    if (prefix.payload_bytes > session_.maximum_payload) {
        throw std::runtime_error("Rune tensor payload exceeds the Harbor safety limit.");
    }
    std::string name(prefix.name_bytes, '\0');
    channel_->read_all(std::as_writable_bytes(std::span(name)));
    std::vector<std::byte> shape_bytes(prefix.shape_bytes);
    channel_->read_all(shape_bytes);
    auto shape = decode_shape(shape_bytes, prefix.dimensions);
    std::vector<std::byte> payload(prefix.payload_bytes);
    channel_->read_all(payload);

    Tensor tensor{
        name,
        prefix.element,
        std::move(shape),
        std::move(payload),
        prefix.exchange_index,
        prefix.physical_time,
        prefix.solver_time_index};
    tensor.view().validate();
    return {RuneKind::tensor, std::move(tensor), prefix.exchange_index, 0};
}

Tensor Harbor::receive() {
    auto message = receive_message();
    if (message.kind != RuneKind::tensor || !message.tensor.has_value()) {
        throw std::runtime_error("Harbor expected a tensor Rune message.");
    }
    return std::move(*message.tensor);
}

void Harbor::send_control(
    RuneKind kind,
    const SessionHello& session,
    std::uint64_t exchange_index,
    std::uint32_t tensor_count) {
    if (kind == RuneKind::hello || kind == RuneKind::hello_accept) {
        session.validate();
    }
    const RunePrefix prefix{
        .kind = kind,
        .flags = kind == RuneKind::complete
                     ? tensor_count
                     : static_cast<std::uint32_t>(session.capabilities),
        .exchange_index = exchange_index,
        .rank = session.rank,
        .peers = session.peers,
        .session_id = session.session_id,
        .maximum_payload = session.maximum_payload,
    };
    channel_->write_all(encode_prefix(prefix));
}

RunePrefix Harbor::receive_prefix() {
    std::array<std::byte, rune_prefix_size> header{};
    channel_->read_all(header);
    return decode_prefix(header);
}

void Harbor::wait_for_handshake() {
    if (options_.handshake == HandshakeMode::timed
        && !channel_->wait_readable(options_.handshake_timeout)) {
        throw std::runtime_error("FoamNordic handshake timed out.");
    }
}

SessionHello Harbor::connect_session(const SessionHello& offer) {
    offer.validate();
    if (options_.handshake == HandshakeMode::disabled) {
        session_ = offer;
        return session_;
    }
    send_control(RuneKind::hello, offer);
    wait_for_handshake();
    const auto reply = receive_prefix();
    if (reply.kind == RuneKind::hello_reject) {
        throw std::runtime_error("The remote Harbor rejected the FoamNordic session.");
    }
    if (reply.kind != RuneKind::hello_accept) {
        throw std::runtime_error("Harbor expected a Rune hello acceptance.");
    }
    SessionHello selected{
        reply.session_id,
        static_cast<Capability>(reply.flags),
        reply.rank,
        reply.peers,
        reply.maximum_payload,
    };
    selected.validate();
    if (selected.session_id != offer.session_id) {
        throw std::runtime_error("Harbor accepted an unexpected session ID.");
    }
    if ((selected.capabilities & offer.capabilities) != selected.capabilities) {
        throw std::runtime_error("Harbor selected a capability that was not offered.");
    }
    if (selected.maximum_payload > offer.maximum_payload) {
        throw std::runtime_error("Harbor selected an invalid payload limit.");
    }
    session_ = selected;
    return session_;
}

SessionHello Harbor::accept_session(const SessionHello& support) {
    support.validate();
    if (options_.handshake == HandshakeMode::disabled) {
        session_ = support;
        return session_;
    }
    wait_for_handshake();
    const auto request = receive_prefix();
    if (request.kind != RuneKind::hello) {
        throw std::runtime_error("Harbor expected a Rune hello request.");
    }
    SessionHello offered{
        request.session_id,
        static_cast<Capability>(request.flags),
        request.rank,
        request.peers,
        request.maximum_payload,
    };
    offered.validate();
    const auto selected_capabilities = offered.capabilities & support.capabilities;
    if (!any(selected_capabilities)) {
        SessionHello rejected = offered;
        rejected.capabilities = Capability::none;
        send_control(RuneKind::hello_reject, rejected);
        throw std::runtime_error("No mutually supported FoamNordic channel capability.");
    }
    session_ = SessionHello{
        offered.session_id,
        selected_capabilities,
        offered.rank,
        offered.peers,
        std::min(offered.maximum_payload, support.maximum_payload),
    };
    send_control(RuneKind::hello_accept, session_);
    return session_;
}

void Harbor::complete(std::uint64_t exchange_index, std::uint32_t tensor_count) {
    if (tensor_count == 0) {
        throw std::invalid_argument("An atomic exchange commit requires a tensor count.");
    }
    send_control(RuneKind::complete, session_, exchange_index, tensor_count);
}

void Harbor::fail_exchange(std::uint64_t exchange_index) {
    send_control(RuneKind::error, session_, exchange_index);
}

void Harbor::shutdown() { send_control(RuneKind::shutdown, session_); }

void Harbor::offer_shared_memory(const std::string& name) {
    if (!any(session_.capabilities & Capability::shm)) {
        throw std::logic_error("Shared memory was not negotiated for this Harbor session.");
    }
    if (name.empty() || name.size() > 4096) {
        throw std::invalid_argument("FoamNordic SHM offer name is invalid.");
    }
    const RunePrefix offer{
        .kind = RuneKind::shm_offer,
        .name_bytes = static_cast<std::uint32_t>(name.size()),
        .session_id = session_.session_id,
    };
    channel_->write_all(encode_prefix(offer));
    channel_->write_all(std::as_bytes(std::span(name)));

    const auto ready = receive_prefix();
    if (ready.kind != RuneKind::shm_ready
        || ready.session_id != session_.session_id) {
        throw std::runtime_error("FoamNordic peer did not complete the SHM upgrade.");
    }
    channel_ = connect_shared_memory_channel(
        name, std::move(channel_), SharedMemoryRole::responder);
}

void Harbor::accept_shared_memory() {
    if (!any(session_.capabilities & Capability::shm)) {
        throw std::logic_error("Shared memory was not negotiated for this Harbor session.");
    }
    const auto offer = receive_prefix();
    if (offer.kind != RuneKind::shm_offer
        || offer.session_id != session_.session_id
        || offer.name_bytes == 0 || offer.name_bytes > 4096) {
        throw std::runtime_error("FoamNordic received an invalid SHM upgrade offer.");
    }
    std::string name(offer.name_bytes, '\0');
    channel_->read_all(std::as_writable_bytes(std::span(name)));
    auto wake_channel = channel_->duplicate();
    auto shared_memory = create_shared_memory_channel(
        name, std::move(wake_channel), SharedMemoryRole::initiator);
    send_control(RuneKind::shm_ready, session_);
    channel_->close();
    channel_ = std::move(shared_memory);
}

#ifdef FOAMNORDIC_HAVE_UCX
void Harbor::offer_ucx(UcxListener& listener) {
    if (!any(session_.capabilities & Capability::ucx)) {
        throw std::logic_error("UCX was not negotiated for this Harbor session.");
    }
    const auto address = listener.address().text();
    if (address.size() > 4096) {
        throw std::invalid_argument("FoamNordic UCX offer address is too long.");
    }
    const RunePrefix offer{
        .kind = RuneKind::ucx_offer,
        .name_bytes = static_cast<std::uint32_t>(address.size()),
        .session_id = session_.session_id,
    };
    channel_->write_all(encode_prefix(offer));
    channel_->write_all(std::as_bytes(std::span(address)));

    auto ucx = listener.accept(options_.handshake_timeout);
    std::array<std::byte, 1> bootstrap{};
    ucx->read_all(bootstrap);
    if (bootstrap.front() != std::byte{0x46}) {
        throw std::runtime_error(
            "FoamNordic peer sent an invalid UCX bootstrap.");
    }
    const auto ready = receive_prefix();
    if (ready.kind != RuneKind::ucx_ready
        || ready.session_id != session_.session_id) {
        throw std::runtime_error("FoamNordic peer did not accept the UCX upgrade.");
    }
    channel_->close();
    channel_ = std::move(ucx);
}

void Harbor::accept_ucx() {
    if (!any(session_.capabilities & Capability::ucx)) {
        throw std::logic_error("UCX was not negotiated for this Harbor session.");
    }
    const auto offer = receive_prefix();
    if (offer.kind != RuneKind::ucx_offer
        || offer.session_id != session_.session_id
        || offer.name_bytes == 0 || offer.name_bytes > 4096) {
        throw std::runtime_error("FoamNordic received an invalid UCX upgrade offer.");
    }
    std::string address_text(offer.name_bytes, '\0');
    channel_->read_all(std::as_writable_bytes(std::span(address_text)));
    auto ucx = connect_ucx(FjordAddress::parse(address_text));
    const std::array bootstrap{std::byte{0x46}};
    ucx->write_all(bootstrap);
    send_control(RuneKind::ucx_ready, session_);
    channel_->close();
    channel_ = std::move(ucx);
}
#endif

RuneKind Harbor::receive_control(std::uint64_t* exchange_index) {
    auto message = receive_message();
    if (message.kind != RuneKind::complete && message.kind != RuneKind::shutdown
        && message.kind != RuneKind::error) {
        throw std::runtime_error("Harbor expected a Rune lifecycle message.");
    }
    if (exchange_index != nullptr) {
        *exchange_index = message.exchange_index;
    }
    return message.kind;
}

void Harbor::interrupt() noexcept { channel_->interrupt(); }

void Harbor::close() noexcept { channel_->close(); }

}  // namespace foamnordic::fjord
