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

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace foamnordic::native {

class LongshipStop {
public:
    void request_stop() noexcept;
    [[nodiscard]] bool stop_requested() const noexcept;

private:
    std::atomic<bool> requested_{false};
};

struct LongshipCommand {
    std::vector<std::string> arguments;
    std::filesystem::path output;

    void validate() const;
};

struct LongshipLaunch {
    LongshipCommand host;
    LongshipCommand solver;
    std::vector<std::filesystem::path> host_ready_files;
    std::chrono::milliseconds readiness_timeout{30'000};
    std::chrono::milliseconds termination_grace{2'000};

    void validate() const;
};

struct LongshipResult {
    int solver_status{0};
    int host_status{0};
    bool host_failed_first{false};
    bool cancelled{false};

    [[nodiscard]] bool success() const noexcept {
        return solver_status == 0 && !host_failed_first && !cancelled;
    }
};

[[nodiscard]] LongshipResult sail_longship(
    const LongshipLaunch& launch,
    const LongshipStop* stop = nullptr);

}  // namespace foamnordic::native
