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

#include "closureSession.H"

#include "Pstream.H"

#include "foamnordic/fjord/endpoint.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace Foam::foamNordic {

string resolveRankAddress(const string& address) {
    std::string resolved(address.c_str());
    const std::string marker("{rank}");
    const auto position = resolved.find(marker);
    if (position != std::string::npos) {
        resolved.replace(
            position,
            marker.size(),
            std::to_string(Pstream::myProcNo()));
    }
    return resolved;
}

std::unique_ptr<foamnordic::fjord::Harbor> connectSession(
    const string& address,
    bool sharedMemory,
    std::uint64_t sessionId) {
    if (address.empty() || sessionId == 0) {
        throw std::invalid_argument(
            "FoamNordic closure address and positive sessionId are required.");
    }
    const auto endpoint = foamnordic::fjord::FjordAddress::parse(
        resolveRankAddress(address).c_str());
    auto harbor = std::make_unique<foamnordic::fjord::Harbor>(
        foamnordic::fjord::connect(endpoint));
    auto capabilities =
        endpoint.kind == foamnordic::fjord::FjordKind::unix_socket
            ? foamnordic::fjord::Capability::uds
            : foamnordic::fjord::Capability::tcp;
    if (sharedMemory
        && endpoint.kind == foamnordic::fjord::FjordKind::unix_socket) {
        capabilities = capabilities | foamnordic::fjord::Capability::shm;
    }
    const auto selected = harbor->connect_session({
        sessionId,
        capabilities,
        static_cast<std::uint32_t>(Pstream::myProcNo()),
        static_cast<std::uint32_t>(Pstream::nProcs()),
        16ULL * 1024ULL * 1024ULL * 1024ULL,
    });
    if (sharedMemory
        && foamnordic::fjord::any(
            selected.capabilities & foamnordic::fjord::Capability::shm)) {
        harbor->accept_shared_memory();
    }
    return harbor;
}

ClosureSession::ClosureSession(
    const dictionary& dict,
    foamnordic::adapter::ExchangeContract contract) {
    const auto address = dict.get<string>("address");
    const auto sharedMemory = dict.getOrDefault<bool>("sharedMemory", true);
    const auto sessionId = static_cast<std::uint64_t>(
        dict.getOrDefault<label>("sessionId", 1));
    harbor_ = connectSession(address, sharedMemory, sessionId);
    port_ = std::make_unique<foamnordic::adapter::ClosurePort>(
        *harbor_, std::move(contract));
}

ClosureSession::~ClosureSession() {
    try {
        shutdown();
    } catch (...) {
    }
}

foamnordic::adapter::ClosureInvocation ClosureSession::begin(
    const Time& time) {
    if (!port_) {
        throw std::logic_error("FoamNordic closure session is shut down.");
    }
    return port_->begin(
        static_cast<std::uint64_t>(time.timeIndex()),
        static_cast<double>(time.value()));
}

void ClosureSession::shutdown() {
    port_.reset();
    if (harbor_) {
        harbor_->shutdown();
        harbor_.reset();
    }
}

}  // namespace Foam::foamNordic
