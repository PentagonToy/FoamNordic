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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"

namespace {

template<class Value>
void scale_values(std::vector<std::byte>& bytes, double scale) {
    for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(Value)) {
        Value value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(Value));
        value = static_cast<Value>(static_cast<double>(value) * scale);
        std::memcpy(bytes.data() + offset, &value, sizeof(Value));
    }
}

void scale_tensor(foamnordic::fjord::Tensor& tensor, double scale) {
    switch (tensor.element) {
        case foamnordic::fjord::Element::float32:
            scale_values<float>(tensor.bytes, scale);
            return;
        case foamnordic::fjord::Element::float64:
            scale_values<double>(tensor.bytes, scale);
            return;
        case foamnordic::fjord::Element::int32:
        case foamnordic::fjord::Element::int64:
            throw std::invalid_argument(
                "Echo scaling requires floating-point tensor storage.");
    }
}

foamnordic::fjord::FjordListener listen(
    const foamnordic::fjord::FjordAddress& address) {
    if (address.kind == foamnordic::fjord::FjordKind::unix_socket) {
        return foamnordic::fjord::FjordListener::local(address.location);
    }
    return foamnordic::fjord::FjordListener::network(
        address.location, address.port);
}

void run(
    const std::string& addressText,
    const std::string& outputName,
    const std::string& sourceName,
    double scale,
    bool rejectExchange) {
    const auto address = foamnordic::fjord::FjordAddress::parse(addressText);
    auto listener = listen(address);
    std::cout << "[FoamNord] Echo worker listening: "
              << listener.address().text() << std::endl;

    foamnordic::fjord::Harbor harbor(listener.accept());
    auto capabilities = address.kind == foamnordic::fjord::FjordKind::unix_socket
                            ? foamnordic::fjord::Capability::uds
                                  | foamnordic::fjord::Capability::shm
                            : foamnordic::fjord::Capability::tcp;
    const auto session = harbor.accept_session({
        1,
        capabilities,
        0,
        1,
        16ULL * 1024ULL * 1024ULL * 1024ULL,
    });
    if (foamnordic::fjord::any(
            session.capabilities & foamnordic::fjord::Capability::shm)) {
        const auto shmName = std::string("/foamnordic-openfoam-")
                             + std::to_string(static_cast<long long>(::getpid()))
                             + '-' + std::to_string(session.session_id);
        harbor.offer_shared_memory(shmName);
        std::cout << "[FoamNord] Data plane: SHM" << std::endl;
    } else {
        std::cout << "[FoamNord] Data plane: "
                  << (address.kind == foamnordic::fjord::FjordKind::unix_socket
                          ? "UDS" : "TCP")
                  << std::endl;
    }

    std::uint64_t expectedExchangeIndex = 0;
    std::optional<std::uint64_t> lastSolverTimeIndex;
    std::optional<double> lastPhysicalTime;
    while (true) {
        std::vector<foamnordic::fjord::Tensor> prepared;
        std::optional<std::uint64_t> exchangeIndex;
        std::optional<std::uint64_t> exchangeSolverTimeIndex;
        std::optional<double> exchangePhysicalTime;
        while (true) {
            auto message = harbor.receive_message();
            if (message.kind == foamnordic::fjord::RuneKind::shutdown) {
                std::cout << "[FoamNord] Echo worker stopped." << std::endl;
                return;
            }
            if (message.kind == foamnordic::fjord::RuneKind::tensor) {
                if (!message.tensor) {
                    throw std::runtime_error(
                        "Echo worker received an empty tensor message.");
                }
                if (!exchangeIndex) {
                    exchangeIndex = message.exchange_index;
                    exchangeSolverTimeIndex =
                        message.tensor->solver_time_index;
                    exchangePhysicalTime = message.tensor->physical_time;
                }
                if (message.exchange_index != *exchangeIndex
                    || message.tensor->time_index != *exchangeIndex
                    || message.tensor->solver_time_index
                           != *exchangeSolverTimeIndex
                    || message.tensor->physical_time != *exchangePhysicalTime) {
                    throw std::runtime_error(
                        "Echo worker received inconsistent exchange metadata.");
                }
                prepared.push_back(std::move(*message.tensor));
                continue;
            }
            if (!exchangeIndex
                || message.kind != foamnordic::fjord::RuneKind::complete
                || message.exchange_index != *exchangeIndex
                || message.tensor_count != prepared.size()) {
                throw std::runtime_error("Echo worker received an incomplete exchange.");
            }
            break;
        }
        if (*exchangeIndex != expectedExchangeIndex) {
            throw std::runtime_error(
                "Echo worker received a non-monotonic closure call index.");
        }
        if (lastSolverTimeIndex) {
            if (*exchangeSolverTimeIndex < *lastSolverTimeIndex
                || *exchangePhysicalTime < *lastPhysicalTime) {
                throw std::runtime_error(
                    "Echo worker received regressing solver-time metadata.");
            }
            if (*exchangeSolverTimeIndex == *lastSolverTimeIndex
                && *exchangePhysicalTime != *lastPhysicalTime) {
                throw std::runtime_error(
                    "Echo worker received inconsistent repeated-call time.");
            }
        }
        lastSolverTimeIndex = *exchangeSolverTimeIndex;
        lastPhysicalTime = *exchangePhysicalTime;
        if (rejectExchange) {
            harbor.fail_exchange(*exchangeIndex);
            std::cout
                << "[FoamNord] Rejected closure call " << *exchangeIndex
                << " at solver time " << *exchangeSolverTimeIndex
                << " at physical time " << *exchangePhysicalTime
                << std::endl;
            return;
        }
        const auto output = std::find_if(
            prepared.begin(), prepared.end(), [&](const auto& tensor) {
                return tensor.name == sourceName;
            });
        if (output == prepared.end()) {
            throw std::runtime_error(
                "Echo source is not present among the input fields: "
                + sourceName);
        }
        auto transformed = *output;
        transformed.name = outputName;
        scale_tensor(transformed, scale);
        const auto view = transformed.view();
        harbor.publish(*exchangeIndex, std::span(&view, 1));
        std::cout
            << "[FoamNord] Same-time closure call " << *exchangeIndex
            << ": solver time " << *exchangeSolverTimeIndex
            << ", physical time " << *exchangePhysicalTime << std::endl;
        ++expectedExchangeIndex;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Usage: foamnordic_openfoam_echo <address> <output-field> "
               "[scale] [--source <input-field>] [--reject]\n";
        return 2;
    }
    try {
        double scale = 1.0;
        std::string sourceName(argv[2]);
        bool rejectExchange = false;
        int argument = 3;
        if (argument < argc
            && std::string(argv[argument]).rfind("--", 0) != 0) {
            scale = std::stod(argv[argument++]);
        }
        while (argument < argc) {
            const std::string option(argv[argument++]);
            if (option == "--reject") {
                rejectExchange = true;
            } else if (option == "--source" && argument < argc) {
                sourceName = argv[argument++];
            } else {
                throw std::invalid_argument(
                    "Unknown or incomplete echo worker option: " + option);
            }
        }
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("Echo scale must be finite.");
        }
        run(argv[1], argv[2], sourceName, scale, rejectExchange);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FoamNord] Echo worker failed: " << error.what() << '\n';
        return 1;
    }
}
