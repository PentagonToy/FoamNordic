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
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

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
    void interrupt() noexcept;
    void close() noexcept;

private:
    std::unique_ptr<fjord::FjordChannel> channel_;
};

struct ObservationSource {
    std::string name;
    std::unique_ptr<ObservationReceiver> receiver;
};

struct LongshipObservation {
    std::uint64_t stream_index{0};
    std::string source;
    ObservationRecord record;

    [[nodiscard]] std::size_t byte_size() const noexcept;
};

class LongshipObservationStream {
public:
    explicit LongshipObservationStream(
        std::vector<ObservationSource> sources,
        ObservationRetention retention = {});
    ~LongshipObservationStream();

    LongshipObservationStream(const LongshipObservationStream&) = delete;
    LongshipObservationStream& operator=(
        const LongshipObservationStream&) = delete;

    [[nodiscard]] std::optional<LongshipObservation> receive(
        std::chrono::milliseconds timeout);
    void stop() noexcept;
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] std::string failure() const;
    [[nodiscard]] std::uint64_t dropped_records() const noexcept;

private:
    void run(std::size_t source_index) noexcept;
    void retain(LongshipObservation observation);

    std::vector<ObservationSource> sources_;
    ObservationRetention retention_;
    std::vector<std::thread> workers_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<LongshipObservation> records_;
    std::size_t bytes_{0};
    std::uint64_t next_stream_index_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<bool> healthy_{true};
    std::atomic<bool> stopped_{false};
    std::string failure_;
};

}  // namespace foamnordic::adapter
