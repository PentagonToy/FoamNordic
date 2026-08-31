#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"
#include "foamnordic/fjord/rune.hpp"
#ifdef FOAMNORDIC_HAVE_UCX
#include "foamnordic/fjord/ucx_channel.hpp"
#endif

using foamnordic::fjord::Element;
using foamnordic::fjord::Harbor;
using foamnordic::fjord::TensorView;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<double> values_of(const foamnordic::fjord::Tensor& tensor) {
    require(
        tensor.bytes.size() % sizeof(double) == 0,
        "Tensor payload is not aligned to float64 values.");
    std::vector<double> values(tensor.bytes.size() / sizeof(double));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

void test_rune_codec() {
    const std::array<double, 6> values{0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    const TensorView original{
        "U", Element::float64, {2, 3}, foamnordic::fjord::as_bytes(std::span(values)),
        17, 0.125, 9};

    const auto frame = foamnordic::fjord::encode_tensor(original);
    const auto decoded = foamnordic::fjord::decode_tensor(frame);

    require(decoded.name == "U", "Rune did not preserve the tensor name.");
    require(decoded.element == Element::float64, "Rune did not preserve the element type.");
    require(
        decoded.shape == std::vector<std::uint64_t>{2, 3},
        "Rune did not preserve the tensor shape.");
    require(decoded.time_index == 17, "Rune did not preserve the time index.");
    require(
        decoded.solver_time_index == 9,
        "Rune did not preserve the solver time index.");
    require(
        std::abs(decoded.physical_time - 0.125) < 1.0e-15,
        "Rune did not preserve physical time.");
    require(
        values_of(decoded) == std::vector<double>(values.begin(), values.end()),
        "Rune did not preserve tensor values.");
}

void test_upgrade_control_codec() {
    for (const auto kind : {
             foamnordic::fjord::RuneKind::shm_offer,
             foamnordic::fjord::RuneKind::ucx_offer}) {
        const foamnordic::fjord::RunePrefix prefix{
            .kind = kind,
            .name_bytes = 24,
            .session_id = 91,
        };
        const auto decoded = foamnordic::fjord::decode_prefix(
            foamnordic::fjord::encode_prefix(prefix));
        require(decoded.kind == kind, "Rune lost an upgrade offer kind.");
        require(
            decoded.name_bytes == 24 && decoded.session_id == 91,
            "Rune lost upgrade offer metadata.");
    }
    for (const auto kind : {
             foamnordic::fjord::RuneKind::shm_ready,
             foamnordic::fjord::RuneKind::ucx_ready}) {
        const foamnordic::fjord::RunePrefix prefix{
            .kind = kind,
            .session_id = 91,
        };
        const auto decoded = foamnordic::fjord::decode_prefix(
            foamnordic::fjord::encode_prefix(prefix));
        require(decoded.kind == kind, "Rune lost an upgrade readiness kind.");
    }
}

void test_harbor_round_trip() {
    auto [client_channel, worker_channel] = foamnordic::fjord::local_channel_pair();
    Harbor client(std::move(client_channel));
    Harbor worker(std::move(worker_channel));

    std::thread exchange([&worker] {
        auto input = worker.receive();
        auto values = values_of(input);
        for (auto& value : values) {
            value *= 2.0;
        }
        const TensorView output{
            input.name,
            Element::float64,
            input.shape,
            foamnordic::fjord::as_bytes(std::span<const double>(values)),
            input.time_index,
            input.physical_time,
        };
        worker.send(output);
    });

    const std::array<double, 6> velocity{0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    const TensorView input{
        "U", Element::float64, {2, 3}, foamnordic::fjord::as_bytes(std::span(velocity)),
        41, 0.25};
    client.send(input);
    const auto output = client.receive();
    exchange.join();

    require(output.name == "U", "Harbor did not preserve the tensor name.");
    require(output.time_index == 41, "Harbor did not preserve the time index.");
    require(
        values_of(output) == std::vector<double>{1.0, 3.0, 5.0, 7.0, 9.0, 11.0},
        "Harbor round-trip values are incorrect.");
}

void test_session_negotiation() {
    auto [client_channel, server_channel] = foamnordic::fjord::local_channel_pair();
    Harbor client(std::move(client_channel));
    Harbor server(std::move(server_channel));

    const foamnordic::fjord::SessionHello offered{
        0x4e4f52444943ULL,
        foamnordic::fjord::Capability::uds | foamnordic::fjord::Capability::tcp,
        1,
        4,
        4096,
    };
    const foamnordic::fjord::SessionHello supported{
        1,
        foamnordic::fjord::Capability::uds | foamnordic::fjord::Capability::shm,
        0,
        1,
        2048,
    };

    std::thread peer([&server, &supported] {
        const auto selected = server.accept_session(supported);
        require(
            selected.capabilities == foamnordic::fjord::Capability::uds,
            "Server selected an incorrect channel capability.");
        require(selected.maximum_payload == 2048, "Server selected an invalid payload limit.");

        std::uint64_t exchange_index = 0;
        require(
            server.receive_control(&exchange_index) == foamnordic::fjord::RuneKind::complete,
            "Server did not receive completion.");
        require(exchange_index == 27, "Completion exchange index is incorrect.");
        require(
            server.receive_control() == foamnordic::fjord::RuneKind::shutdown,
            "Server did not receive shutdown.");
    });

    const auto selected = client.connect_session(offered);
    require(
        selected.capabilities == foamnordic::fjord::Capability::uds,
        "Client selected an incorrect channel capability.");
    require(selected.session_id == offered.session_id, "Session ID was not preserved.");
    require(selected.rank == 1 && selected.peers == 4, "MPI identity was not preserved.");
    require(selected.maximum_payload == 2048, "Client payload limit is incorrect.");
    client.complete(27, 1);
    client.shutdown();
    peer.join();
}

void test_session_rejection() {
    auto [client_channel, server_channel] = foamnordic::fjord::local_channel_pair();
    Harbor client(std::move(client_channel));
    Harbor server(std::move(server_channel));
    std::atomic<bool> server_rejected{false};

    std::thread peer([&server, &server_rejected] {
        try {
            static_cast<void>(server.accept_session({
                1,
                foamnordic::fjord::Capability::uds,
                0,
                1,
                4096,
            }));
        } catch (const std::runtime_error&) {
            server_rejected = true;
        }
    });

    bool client_rejected = false;
    try {
        static_cast<void>(client.connect_session({
            77,
            foamnordic::fjord::Capability::tcp,
            0,
            1,
            4096,
        }));
    } catch (const std::runtime_error&) {
        client_rejected = true;
    }
    peer.join();
    require(client_rejected, "Client accepted a session without a common capability.");
    require(server_rejected, "Server accepted a session without a common capability.");
}

void test_timed_handshake() {
    auto [client_channel, silent_channel] = foamnordic::fjord::local_channel_pair();
    Harbor client(
        std::move(client_channel),
        {
            foamnordic::fjord::HandshakeMode::timed,
            std::chrono::milliseconds(20),
        });

    bool timed_out = false;
    try {
        static_cast<void>(client.connect_session({
            101,
            foamnordic::fjord::Capability::uds,
            0,
            1,
            4096,
        }));
    } catch (const std::runtime_error&) {
        timed_out = true;
    }
    require(timed_out, "Timed handshake did not report an unresponsive peer.");
    silent_channel->close();
}

void exchange_once(foamnordic::fjord::FjordListener& listener) {
    Harbor worker(listener.accept());
    const auto input = worker.receive();
    worker.send(input.view());
}

void verify_endpoint_round_trip(const foamnordic::fjord::FjordAddress& address) {
    const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    Harbor client(foamnordic::fjord::connect(address));
    client.send(TensorView{
        "p",
        Element::float32,
        {4},
        foamnordic::fjord::as_bytes(std::span(values)),
        9,
        0.5,
    });
    const auto output = client.receive();
    require(output.name == "p", "Endpoint did not preserve the field name.");
    require(output.element == Element::float32, "Endpoint did not preserve float32.");
    require(output.shape == std::vector<std::uint64_t>{4}, "Endpoint shape is incorrect.");
    require(output.bytes.size() == sizeof(values), "Endpoint payload size is incorrect.");
    require(
        std::memcmp(output.bytes.data(), values.data(), sizeof(values)) == 0,
        "Endpoint payload is incorrect.");
}

void test_unix_endpoint() {
    const auto path = std::string("/tmp/foamnordic-fjord-")
                      + std::to_string(static_cast<long long>(::getpid())) + ".sock";
    auto listener = foamnordic::fjord::FjordListener::local(path);
    require(
        listener.address().text() == "unix://" + path,
        "Unix endpoint text is incorrect.");
    std::thread server([&listener] { exchange_once(listener); });
    verify_endpoint_round_trip(listener.address());
    server.join();
    listener.close();
    require(::access(path.c_str(), F_OK) != 0, "Unix endpoint file was not cleaned up.");
}

void test_tcp_endpoint() {
    auto listener = foamnordic::fjord::FjordListener::network("127.0.0.1", 0);
    require(listener.address().port != 0, "TCP listener did not resolve its dynamic port.");
    std::thread server([&listener] { exchange_once(listener); });
    verify_endpoint_round_trip(listener.address());
    server.join();
}

void test_address_parser() {
    const auto local = foamnordic::fjord::FjordAddress::parse(
        "unix:///tmp/foamnordic.sock");
    require(
        local.kind == foamnordic::fjord::FjordKind::unix_socket
            && local.location == "/tmp/foamnordic.sock",
        "Unix Fjord address parsing failed.");
    const auto network = foamnordic::fjord::FjordAddress::parse(
        "tcp://127.0.0.1:2026");
    require(
        network.kind == foamnordic::fjord::FjordKind::tcp
            && network.location == "127.0.0.1" && network.port == 2026,
        "TCP Fjord address parsing failed.");
    const auto ucx = foamnordic::fjord::FjordAddress::parse(
        "ucx://127.0.0.1:2027");
    require(
        ucx.kind == foamnordic::fjord::FjordKind::ucx
            && ucx.location == "127.0.0.1" && ucx.port == 2027,
        "UCX Fjord address parsing failed.");
}

#ifdef FOAMNORDIC_HAVE_UCX
void test_ucx_upgrade() {
    auto control = foamnordic::fjord::FjordListener::network("127.0.0.1", 0);
    std::atomic<bool> upgraded{false};
    std::thread server([&control, &upgraded] {
        Harbor harbor(control.accept());
        const auto selected = harbor.accept_session({
            1,
            foamnordic::fjord::Capability::tcp
                | foamnordic::fjord::Capability::ucx,
            0,
            2,
            4096,
        });
        require(
            foamnordic::fjord::any(
                selected.capabilities & foamnordic::fjord::Capability::ucx),
            "UCX loopback session did not select UCX.");
        auto listener = foamnordic::fjord::UcxListener::network("127.0.0.1", 0);
        harbor.offer_ucx(listener);
        upgraded.store(true, std::memory_order_release);
        const auto input = harbor.receive();
        harbor.send(input.view());
    });

    Harbor client(foamnordic::fjord::connect(control.address()));
    const auto selected = client.connect_session({
        87,
        foamnordic::fjord::Capability::tcp
            | foamnordic::fjord::Capability::ucx,
        1,
        2,
        4096,
    });
    require(
        foamnordic::fjord::any(
            selected.capabilities & foamnordic::fjord::Capability::ucx),
        "UCX loopback client did not select UCX.");
    client.accept_ucx();
    const auto upgrade_deadline = std::chrono::steady_clock::now()
                                  + std::chrono::seconds(2);
    while (!upgraded.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < upgrade_deadline) {
        std::this_thread::yield();
    }
    require(
        upgraded.load(std::memory_order_acquire),
        "UCX upgrade remained dependent on the first payload.");
    const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    client.send(TensorView{
        "p",
        Element::float32,
        {4},
        foamnordic::fjord::as_bytes(std::span(values)),
        9,
        0.5,
    });
    const auto output = client.receive();
    server.join();
    require(output.name == "p", "UCX loopback lost the field name.");
    require(
        output.bytes.size() == sizeof(values)
            && std::memcmp(output.bytes.data(), values.data(), sizeof(values)) == 0,
        "UCX loopback corrupted the field payload.");
}
#endif

void verify_interrupt_releases_blocked_reader(
    std::pair<
        std::unique_ptr<foamnordic::fjord::FjordChannel>,
        std::unique_ptr<foamnordic::fjord::FjordChannel>> channels) {
    std::atomic<bool> released{false};
    std::thread reader([&] {
        std::array<std::byte, 8> bytes{};
        try {
            channels.first->read_all(bytes);
        } catch (const std::runtime_error&) {
            released.store(true, std::memory_order_release);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    channels.first->interrupt();
    reader.join();
    channels.first->close();
    channels.second->close();
    require(
        released.load(std::memory_order_acquire),
        "Fjord interrupt did not release a blocked reader.");
}

void test_channel_interrupt() {
    verify_interrupt_releases_blocked_reader(
        foamnordic::fjord::local_channel_pair());
    verify_interrupt_releases_blocked_reader(
        foamnordic::fjord::shared_memory_channel_pair(2, 64));
}

}  // namespace

int main() {
    test_rune_codec();
    test_upgrade_control_codec();
    test_harbor_round_trip();
    test_session_negotiation();
    test_session_rejection();
    test_timed_handshake();
    test_unix_endpoint();
    test_tcp_endpoint();
    test_address_parser();
#ifdef FOAMNORDIC_HAVE_UCX
    test_ucx_upgrade();
#endif
    test_channel_interrupt();
    std::cout << "FoamNordic Fjord round trip: PASS\n";
}
