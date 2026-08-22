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

#include "foamnordic/backend/adapter/observation.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace foamnordic::adapter {

void ObservationSchedule::validate() const {
    if (every == 0) {
        throw std::invalid_argument("FoamNordic observation cadence must be positive.");
    }
}

bool ObservationSchedule::due(std::uint64_t exchange_index) const noexcept {
    return exchange_index >= offset && (exchange_index - offset) % every == 0;
}

std::size_t ObservationRecord::byte_size() const noexcept {
    std::size_t bytes = sizeof(ObservationRecord);
    for (const auto& field : fields) {
        bytes += sizeof(FieldObservation) + field.field.size();
    }
    return bytes;
}

ObservationPlan& ObservationPlan::observe(
    std::string field, ObservationSchedule schedule) {
    if (field.empty()) {
        throw std::invalid_argument("FoamNordic observation field must not be empty.");
    }
    schedule.validate();
    for (const auto& rule : rules_) {
        if (rule.field == field) {
            throw std::invalid_argument(
                "FoamNordic observation plan contains field twice: " + field);
        }
    }
    rules_.push_back({std::move(field), schedule});
    return *this;
}

CompiledObservationPlan ObservationPlan::compile() const {
    if (rules_.empty()) {
        throw std::logic_error("Cannot compile an empty FoamNordic observation plan.");
    }
    return CompiledObservationPlan(rules_);
}

CompiledObservationPlan::CompiledObservationPlan(
    std::vector<ObservationPlan::Rule> rules)
    : rules_(std::move(rules)) {}

std::optional<ObservationRecord> CompiledObservationPlan::execute(
    std::uint64_t exchange_index,
    double physical_time,
    const ReadOnlyFieldMap& fields) const {
    if (!std::isfinite(physical_time)) {
        throw std::invalid_argument("FoamNordic observation time is invalid.");
    }
    ObservationRecord record{exchange_index, physical_time, {}};
    for (const auto& rule : rules_) {
        if (!rule.schedule.due(exchange_index)) {
            continue;
        }
        const auto found = fields.find(rule.field);
        if (found == fields.end()) {
            throw std::invalid_argument(
                "FoamNordic observation plan is missing field: " + rule.field);
        }
        const auto& field = found->second;
        if (field.time_index != exchange_index
            || std::abs(field.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument("FoamNordic observation metadata is out of sync.");
        }
        record.fields.push_back({rule.field, statistics(field)});
    }
    if (record.fields.empty()) {
        return std::nullopt;
    }
    return record;
}

std::size_t CompiledObservationPlan::size() const noexcept { return rules_.size(); }

void ObservationRetention::validate() const {
    if (max_records == 0 || max_bytes == 0) {
        throw std::invalid_argument("FoamNordic observation retention must be positive.");
    }
}

ObservationBuffer::ObservationBuffer(ObservationRetention retention)
    : retention_(retention) {
    retention_.validate();
}

bool ObservationBuffer::try_publish(ObservationRecord record) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (closed_) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto record_bytes = record.byte_size();
    if (record_bytes > retention_.max_bytes) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (retention_.overflow == ObservationOverflow::drop_newest
        && (records_.size() >= retention_.max_records
            || bytes_ + record_bytes > retention_.max_bytes)) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    while (!records_.empty()
           && (records_.size() >= retention_.max_records
               || bytes_ + record_bytes > retention_.max_bytes)) {
        bytes_ -= records_.front().byte_size();
        records_.pop_front();
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    bytes_ += record_bytes;
    records_.push_back(std::move(record));
    lock.unlock();
    available_.notify_one();
    return true;
}

std::optional<ObservationRecord> ObservationBuffer::try_pop_oldest() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || records_.empty()) {
        return std::nullopt;
    }
    auto record = std::move(records_.front());
    bytes_ -= record.byte_size();
    records_.pop_front();
    return record;
}

std::optional<ObservationRecord> ObservationBuffer::wait_pop_oldest() {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this] { return closed_ || !records_.empty(); });
    if (records_.empty()) {
        return std::nullopt;
    }
    auto record = std::move(records_.front());
    bytes_ -= record.byte_size();
    records_.pop_front();
    return record;
}

void ObservationBuffer::close() noexcept {
    {
        std::scoped_lock lock(mutex_);
        closed_ = true;
    }
    available_.notify_all();
}

std::size_t ObservationBuffer::buffered_records() const {
    std::scoped_lock lock(mutex_);
    return records_.size();
}

std::size_t ObservationBuffer::buffered_bytes() const {
    std::scoped_lock lock(mutex_);
    return bytes_;
}

std::uint64_t ObservationBuffer::dropped_records() const {
    return dropped_.load(std::memory_order_relaxed);
}

}  // namespace foamnordic::adapter
