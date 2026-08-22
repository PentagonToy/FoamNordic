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

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"
#include "foamnordic/fjord/tensor.hpp"
#ifdef FOAMNORDIC_HAVE_UCX
#include "foamnordic/fjord/ucx_channel.hpp"
#endif

namespace {

using foamnordic::fjord::Capability;
using foamnordic::fjord::Element;
using foamnordic::fjord::FjordAddress;
using foamnordic::fjord::FjordListener;
using foamnordic::fjord::Harbor;
using foamnordic::fjord::RuneKind;
using foamnordic::fjord::Tensor;
using foamnordic::fjord::TensorView;

struct Exchange {
    Tensor tensor;
    std::uint64_t index{0};
};

[[nodiscard]] Exchange receive_exchange(Harbor& harbor) {
    auto tensor_message = harbor.receive_message();
    if (tensor_message.kind != RuneKind::tensor || !tensor_message.tensor) {
        throw std::runtime_error("Network probe expected one tensor.");
    }
    auto complete_message = harbor.receive_message();
    if (complete_message.kind != RuneKind::complete
        || complete_message.exchange_index != tensor_message.exchange_index
        || complete_message.tensor_count != 1) {
        throw std::runtime_error(
            "Network probe received an incomplete atomic exchange.");
    }
    return {std::move(*tensor_message.tensor), tensor_message.exchange_index};
}

void validate_tensor(
    const Tensor& tensor,
    std::uint64_t exchange_index,
    std::size_t elements) {
    if (tensor.name != "closure_payload"
        || tensor.element != Element::float64
        || tensor.shape != std::vector<std::uint64_t>{elements}
        || tensor.bytes.size() != elements * sizeof(double)
        || tensor.time_index != exchange_index
        || tensor.solver_time_index != exchange_index) {
        throw std::runtime_error("Network probe tensor metadata is invalid.");
    }

    const auto expected_first = static_cast<double>(exchange_index);
    const auto expected_last = expected_first
                               + static_cast<double>(elements - 1) * 0.001;
    double first{};
    double last{};
    std::memcpy(&first, tensor.bytes.data(), sizeof(double));
    std::memcpy(
        &last,
        tensor.bytes.data() + (elements - 1) * sizeof(double),
        sizeof(double));
    if (std::abs(first - expected_first) > 1.0e-12
        || std::abs(last - expected_last) > 1.0e-12) {
        throw std::runtime_error("Network probe tensor payload is invalid.");
    }
}

void run_server(
    const FjordAddress& address,
    std::uint64_t iterations,
    std::size_t elements,
    bool use_ucx) {
    auto listener = FjordListener::network(address.location, address.port);
    std::cout << "[FoamNord] Inter-node server listening: "
              << listener.address().text() << std::endl;

    Harbor harbor(listener.accept());
    auto supported = Capability::tcp;
#ifdef FOAMNORDIC_HAVE_UCX
    if (use_ucx) {
        supported = supported | Capability::ucx;
    }
#endif
    const auto session = harbor.accept_session({
        1, supported, 0, 2, 16ULL * 1024ULL * 1024ULL * 1024ULL});
    if (!any(session.capabilities & Capability::tcp)) {
        throw std::runtime_error("Inter-node probe lost its TCP control plane.");
    }
#ifdef FOAMNORDIC_HAVE_UCX
    std::unique_ptr<foamnordic::fjord::UcxListener> ucx_listener;
    if (use_ucx) {
        if (!any(session.capabilities & Capability::ucx)) {
            throw std::runtime_error("Inter-node probe did not negotiate UCX.");
        }
        const auto* selected_host = std::getenv("FOAMNORDIC_UCX_HOST");
        const auto ucx_host = selected_host != nullptr && *selected_host != '\0'
                                  ? std::string(selected_host)
                                  : address.location;
        ucx_listener = std::make_unique<foamnordic::fjord::UcxListener>(
            foamnordic::fjord::UcxListener::network(ucx_host, 0));
        std::cout << "[FoamNord] UCX listener: "
                  << ucx_listener->address().text() << std::endl;
        harbor.offer_ucx(*ucx_listener);
    }
#else
    if (use_ucx) {
        throw std::runtime_error(
            "Inter-node probe was built without FoamNordic UCX support.");
    }
#endif

    for (std::uint64_t index = 0; index < iterations; ++index) {
        auto exchange = receive_exchange(harbor);
        if (exchange.index != index) {
            throw std::runtime_error(
                "Inter-node exchange index is not monotonic.");
        }
        validate_tensor(exchange.tensor, index, elements);
        const auto view = exchange.tensor.view();
        harbor.publish(index, std::span(&view, 1));
    }
    if (harbor.receive_control() != RuneKind::shutdown) {
        throw std::runtime_error("Inter-node client did not shut down cleanly.");
    }
    std::cout << "[FoamNord] Inter-node server: PASS" << std::endl;
}

void run_client(
    const FjordAddress& address,
    std::uint64_t iterations,
    std::size_t elements,
    bool use_ucx) {
    Harbor harbor(foamnordic::fjord::connect(address));
    auto offered = Capability::tcp;
#ifdef FOAMNORDIC_HAVE_UCX
    if (use_ucx) {
        offered = offered | Capability::ucx;
    }
#endif
    const auto session = harbor.connect_session({
        1, offered, 1, 2, 16ULL * 1024ULL * 1024ULL * 1024ULL});
    if (!any(session.capabilities & Capability::tcp)) {
        throw std::runtime_error("Inter-node probe lost its TCP control plane.");
    }
#ifdef FOAMNORDIC_HAVE_UCX
    if (use_ucx) {
        if (!any(session.capabilities & Capability::ucx)) {
            throw std::runtime_error("Inter-node probe did not negotiate UCX.");
        }
        harbor.accept_ucx();
    }
#else
    if (use_ucx) {
        throw std::runtime_error(
            "Inter-node probe was built without FoamNordic UCX support.");
    }
#endif

    std::vector<double> values(elements);
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
        for (std::size_t element = 0; element < elements; ++element) {
            values[element] = static_cast<double>(index)
                              + static_cast<double>(element) * 0.001;
        }
        const TensorView tensor{
            "closure_payload",
            Element::float64,
            {elements},
            foamnordic::fjord::as_bytes(std::span<const double>(values)),
            index,
            static_cast<double>(index) * 0.001,
            index,
        };
        harbor.publish(index, std::span(&tensor, 1));
        auto exchange = receive_exchange(harbor);
        if (exchange.index != index) {
            throw std::runtime_error(
                "Inter-node response index is not monotonic.");
        }
        validate_tensor(exchange.tensor, index, elements);
    }
    harbor.shutdown();
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto transferred = static_cast<double>(
        iterations * elements * sizeof(double) * 2ULL);

    std::cout << "[FoamNord] Data plane: " << (use_ucx ? "UCX" : "TCP") << '\n'
              << "[FoamNord] Atomic exchanges: " << iterations << '\n'
              << "[FoamNord] Payload bytes/exchange: "
              << elements * sizeof(double) << '\n'
              << "[FoamNord] Round trips/s: "
              << static_cast<double>(iterations) / elapsed << '\n'
              << "[FoamNord] Payload MiB/s: "
              << transferred / elapsed / (1024.0 * 1024.0) << '\n'
              << "[FoamNord] Inter-node client: PASS" << std::endl;
}

[[nodiscard]] std::uint64_t parse_positive(
    const char* text,
    const char* description) {
    const auto value = std::stoull(text);
    if (value == 0) {
        throw std::invalid_argument(
            std::string(description) + " must be positive.");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr
            << "Usage: foamnordic_fjord_network_probe <server|client> "
               "<tcp-address> <iterations> <elements> [tcp|ucx]\n";
        return 2;
    }
    try {
        const std::string mode(argv[1]);
        const auto address = FjordAddress::parse(argv[2]);
        if (address.kind != foamnordic::fjord::FjordKind::tcp) {
            throw std::invalid_argument("Network probe requires a TCP address.");
        }
        const auto iterations = parse_positive(argv[3], "Iterations");
        const auto elements = static_cast<std::size_t>(
            parse_positive(argv[4], "Elements"));
        const std::string transport = argc == 6 ? argv[5] : "tcp";
        if (transport != "tcp" && transport != "ucx") {
            throw std::invalid_argument("Network probe transport is invalid.");
        }
        const auto use_ucx = transport == "ucx";
        if (mode == "server") {
            run_server(address, iterations, elements, use_ucx);
        } else if (mode == "client") {
            run_client(address, iterations, elements, use_ucx);
        } else {
            throw std::invalid_argument("Network probe mode is invalid.");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FoamNord] Inter-node probe failed: "
                  << error.what() << '\n';
        return 1;
    }
}
