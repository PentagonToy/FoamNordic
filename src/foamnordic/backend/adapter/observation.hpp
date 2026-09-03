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
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "foamnordic/backend/adapter/field.hpp"

namespace foamnordic::adapter {

struct ObservationSchedule {
    std::uint64_t every{1};
    std::uint64_t offset{0};

    void validate() const;
    [[nodiscard]] bool due(std::uint64_t exchange_index) const noexcept;
};

struct FieldObservation {
    std::string field;
    FieldStatistics values;
};

struct ObservationRecord {
    std::uint64_t exchange_index{0};
    double physical_time{0.0};
    std::vector<FieldObservation> fields;
    double model_wait{0.0};
    double evaluate{0.0};

    [[nodiscard]] std::size_t byte_size() const noexcept;
};

using ReadOnlyFieldMap = std::unordered_map<std::string, fjord::TensorView>;

class CompiledObservationPlan;

class ObservationPlan {
public:
    ObservationPlan& observe(std::string field, ObservationSchedule schedule = {});
    [[nodiscard]] CompiledObservationPlan compile() const;

private:
    struct Rule {
        std::string field;
        ObservationSchedule schedule;
    };
    std::vector<Rule> rules_;

    friend class CompiledObservationPlan;
};

class CompiledObservationPlan {
public:
    [[nodiscard]] std::optional<ObservationRecord> execute(
        std::uint64_t exchange_index,
        double physical_time,
        const ReadOnlyFieldMap& fields) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    explicit CompiledObservationPlan(std::vector<ObservationPlan::Rule> rules);
    std::vector<ObservationPlan::Rule> rules_;

    friend class ObservationPlan;
};

enum class ObservationOverflow { drop_oldest, drop_newest };

struct ObservationRetention {
    std::size_t max_records{64};
    std::size_t max_bytes{256U * 1024U};
    ObservationOverflow overflow{ObservationOverflow::drop_oldest};

    void validate() const;
};

class ObservationBuffer {
public:
    explicit ObservationBuffer(ObservationRetention retention = {});

    // Deliberately never waits for a consumer or another publisher.
    [[nodiscard]] bool try_publish(ObservationRecord record);
    [[nodiscard]] std::optional<ObservationRecord> try_pop_oldest();
    [[nodiscard]] std::optional<ObservationRecord> wait_pop_oldest();
    void close() noexcept;
    [[nodiscard]] std::size_t buffered_records() const;
    [[nodiscard]] std::size_t buffered_bytes() const;
    [[nodiscard]] std::uint64_t dropped_records() const;

private:
    ObservationRetention retention_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<ObservationRecord> records_;
    std::size_t bytes_{0};
    std::atomic<std::uint64_t> dropped_{0};
    bool closed_{false};
};

class ObservationJsonlWriter {
public:
    explicit ObservationJsonlWriter(
        std::filesystem::path path,
        ObservationRetention retention = {});
    ~ObservationJsonlWriter();

    ObservationJsonlWriter(const ObservationJsonlWriter&) = delete;
    ObservationJsonlWriter& operator=(const ObservationJsonlWriter&) = delete;

    [[nodiscard]] bool try_publish(ObservationRecord record);
    void stop() noexcept;
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] std::string failure() const;
    [[nodiscard]] std::uint64_t dropped_records() const;

private:
    void run() noexcept;

    std::filesystem::path path_;
    ObservationBuffer buffer_;
    std::thread worker_;
    std::atomic<bool> healthy_{true};
    std::atomic<bool> stopped_{false};
    mutable std::mutex failure_mutex_;
    std::string failure_;
};

}  // namespace foamnordic::adapter
