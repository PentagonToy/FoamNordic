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

#ifndef Foam_foamNordic_closureSession_H
#define Foam_foamNordic_closureSession_H

#include "Time.H"
#include "dictionary.H"
#include "string.H"

#include "foamnordic/backend/adapter/port.hpp"

#include <cstdint>
#include <memory>

namespace foamnordic::fjord {
class Harbor;
}

namespace Foam::foamNordic {

[[nodiscard]] string resolveRankAddress(const string& address);

[[nodiscard]] std::unique_ptr<foamnordic::fjord::Harbor> connectSession(
    const string& address,
    bool sharedMemory,
    bool ucx,
    std::uint64_t sessionId);

class ClosureSession final {
    std::unique_ptr<foamnordic::fjord::Harbor> harbor_;
    std::unique_ptr<foamnordic::adapter::ClosurePort> port_;

public:
    ClosureSession(
        const dictionary& dict,
        foamnordic::adapter::ExchangeContract contract);
    ~ClosureSession();

    ClosureSession(const ClosureSession&) = delete;
    ClosureSession& operator=(const ClosureSession&) = delete;

    [[nodiscard]] foamnordic::adapter::ClosureInvocation begin(
        const Time& time);

    void shutdown();
};

}  // namespace Foam::foamNordic

#endif
