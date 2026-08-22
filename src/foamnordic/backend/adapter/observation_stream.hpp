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
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "foamnordic/backend/adapter/observation.hpp"
#include "foamnordic/fjord/channel.hpp"

namespace foamnordic::adapter {

class ObservationPublisher {
public:
    ObservationPublisher(
        std::unique_ptr<fjord::FjordChannel> channel,
        ObservationRetention retention = {});
    ~ObservationPublisher();

    ObservationPublisher(const ObservationPublisher&) = delete;
    ObservationPublisher& operator=(const ObservationPublisher&) = delete;

    [[nodiscard]] bool try_publish(ObservationRecord record);
    void stop() noexcept;
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] std::string failure() const;
    [[nodiscard]] std::uint64_t dropped_records() const;

private:
    void run() noexcept;

    std::unique_ptr<fjord::FjordChannel> channel_;
    ObservationBuffer buffer_;
    std::thread worker_;
    std::atomic<bool> healthy_{true};
    std::atomic<bool> stopped_{false};
    mutable std::mutex failure_mutex_;
    std::string failure_;
};

class ObservationReceiver {
public:
    explicit ObservationReceiver(std::unique_ptr<fjord::FjordChannel> channel);

    [[nodiscard]] std::optional<ObservationRecord> receive(
        std::chrono::milliseconds timeout);
    void close() noexcept;

private:
    std::unique_ptr<fjord::FjordChannel> channel_;
};

}  // namespace foamnordic::adapter
