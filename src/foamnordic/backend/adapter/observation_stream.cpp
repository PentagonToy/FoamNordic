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

#include "foamnordic/backend/adapter/observation_stream.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace foamnordic::adapter {
namespace {

constexpr std::array<std::byte, 4> mark{
    std::byte{'F'}, std::byte{'N'}, std::byte{'O'}, std::byte{'B'}};
constexpr std::uint16_t version = 2;
constexpr std::size_t header_size = 48;
constexpr std::size_t field_size = 48;
constexpr std::uint64_t maximum_frame = 16U * 1024U * 1024U;

template <class Integer>
void write_integer(std::span<std::byte> bytes, std::size_t offset, Integer value) {
    static_assert(std::is_unsigned_v<Integer>);
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        bytes[offset + index] = std::byte((value >> (index * 8U)) & 0xffU);
    }
}

template <class Integer>
[[nodiscard]] Integer read_integer(
    std::span<const std::byte> bytes, std::size_t offset) {
    static_assert(std::is_unsigned_v<Integer>);
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        value |= static_cast<Integer>(
                     std::to_integer<unsigned char>(bytes[offset + index]))
                 << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::vector<std::byte> encode(const ObservationRecord& record) {
    if (record.fields.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("FoamNordic observation has too many fields.");
    }
    std::size_t total = header_size;
    for (const auto& field : record.fields) {
        if (field.field.empty()
            || field.field.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("FoamNordic observation field name is invalid.");
        }
        total += field_size + field.field.size();
    }
    if (total > maximum_frame) {
        throw std::length_error("FoamNordic observation frame exceeds its limit.");
    }

    std::vector<std::byte> frame(total);
    std::copy(mark.begin(), mark.end(), frame.begin());
    write_integer<std::uint16_t>(frame, 4, version);
    write_integer<std::uint32_t>(
        frame, 8, static_cast<std::uint32_t>(record.fields.size()));
    write_integer<std::uint64_t>(frame, 16, record.exchange_index);
    write_integer<std::uint64_t>(
        frame, 24, std::bit_cast<std::uint64_t>(record.physical_time));
    write_integer<std::uint64_t>(
        frame, 32, std::bit_cast<std::uint64_t>(record.closure_wait));
    write_integer<std::uint64_t>(
        frame, 40, std::bit_cast<std::uint64_t>(record.evaluate));
    std::size_t offset = header_size;
    for (const auto& field : record.fields) {
        write_integer<std::uint32_t>(
            frame, offset, static_cast<std::uint32_t>(field.field.size()));
        write_integer<std::uint64_t>(
            frame, offset + 8, std::bit_cast<std::uint64_t>(field.values.minimum));
        write_integer<std::uint64_t>(
            frame, offset + 16, std::bit_cast<std::uint64_t>(field.values.maximum));
        write_integer<std::uint64_t>(
            frame, offset + 24, std::bit_cast<std::uint64_t>(field.values.mean));
        write_integer<std::uint64_t>(frame, offset + 32, field.values.count);
        write_integer<std::uint64_t>(
            frame, offset + 40, std::bit_cast<std::uint64_t>(field.values.l2));
        offset += field_size;
        const auto name = std::as_bytes(std::span(field.field));
        std::copy(name.begin(), name.end(), frame.begin() + offset);
        offset += name.size();
    }
    return frame;
}

[[nodiscard]] ObservationRecord decode(std::span<const std::byte> frame) {
    if (frame.size() < header_size
        || !std::equal(mark.begin(), mark.end(), frame.begin())
        || read_integer<std::uint16_t>(frame, 4) != version) {
        throw std::runtime_error("Invalid FoamNordic observation frame.");
    }
    ObservationRecord record;
    const auto count = read_integer<std::uint32_t>(frame, 8);
    record.exchange_index = read_integer<std::uint64_t>(frame, 16);
    record.physical_time = std::bit_cast<double>(
        read_integer<std::uint64_t>(frame, 24));
    record.closure_wait = std::bit_cast<double>(
        read_integer<std::uint64_t>(frame, 32));
    record.evaluate = std::bit_cast<double>(
        read_integer<std::uint64_t>(frame, 40));
    record.fields.reserve(count);
    std::size_t offset = header_size;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (frame.size() - offset < field_size) {
            throw std::runtime_error("Truncated FoamNordic observation field.");
        }
        const auto name_size = read_integer<std::uint32_t>(frame, offset);
        FieldStatistics values{
            std::bit_cast<double>(read_integer<std::uint64_t>(frame, offset + 8)),
            std::bit_cast<double>(read_integer<std::uint64_t>(frame, offset + 16)),
            std::bit_cast<double>(read_integer<std::uint64_t>(frame, offset + 24)),
            read_integer<std::uint64_t>(frame, offset + 32),
            std::bit_cast<double>(read_integer<std::uint64_t>(frame, offset + 40)),
        };
        offset += field_size;
        if (name_size == 0 || name_size > frame.size() - offset) {
            throw std::runtime_error("Invalid FoamNordic observation field name.");
        }
        std::string name(name_size, '\0');
        std::copy_n(
            reinterpret_cast<const char*>(frame.data() + offset),
            name_size,
            name.begin());
        offset += name_size;
        record.fields.push_back({std::move(name), values});
    }
    if (offset != frame.size()) {
        throw std::runtime_error("FoamNordic observation frame has trailing data.");
    }
    return record;
}

void write_frame(fjord::FjordChannel& channel, const ObservationRecord& record) {
    const auto frame = encode(record);
    std::array<std::byte, sizeof(std::uint64_t)> length{};
    write_integer<std::uint64_t>(length, 0, frame.size());
    channel.write_all(length);
    channel.write_all(frame);
}

[[nodiscard]] ObservationRecord read_frame(fjord::FjordChannel& channel) {
    std::array<std::byte, sizeof(std::uint64_t)> length{};
    channel.read_all(length);
    const auto size = read_integer<std::uint64_t>(length, 0);
    if (size < header_size || size > maximum_frame) {
        throw std::runtime_error("FoamNordic observation frame length is invalid.");
    }
    std::vector<std::byte> frame(size);
    channel.read_all(frame);
    return decode(frame);
}

}  // namespace

ObservationPublisher::ObservationPublisher(
    std::unique_ptr<fjord::FjordChannel> channel,
    ObservationRetention retention)
    : channel_(std::move(channel)), buffer_(retention) {
    if (!channel_) {
        throw std::invalid_argument("FoamNordic observation publisher requires a channel.");
    }
    worker_ = std::thread([this] { run(); });
}

ObservationPublisher::~ObservationPublisher() { stop(); }

bool ObservationPublisher::try_publish(ObservationRecord record) {
    if (stopped_.load(std::memory_order_acquire)
        || !healthy_.load(std::memory_order_acquire)) {
        return false;
    }
    return buffer_.try_publish(std::move(record));
}

void ObservationPublisher::stop() noexcept {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    buffer_.close();
    channel_->interrupt();
    if (worker_.joinable()) {
        worker_.join();
    }
    channel_->close();
}

bool ObservationPublisher::healthy() const noexcept {
    return healthy_.load(std::memory_order_acquire);
}

std::string ObservationPublisher::failure() const {
    std::scoped_lock lock(failure_mutex_);
    return failure_;
}

std::uint64_t ObservationPublisher::dropped_records() const {
    return buffer_.dropped_records();
}

void ObservationPublisher::run() noexcept {
    try {
        while (auto record = buffer_.wait_pop_oldest()) {
            write_frame(*channel_, *record);
        }
    } catch (const std::exception& error) {
        {
            std::scoped_lock lock(failure_mutex_);
            failure_ = error.what();
        }
        healthy_.store(false, std::memory_order_release);
        buffer_.close();
        channel_->close();
    }
}

ObservationReceiver::ObservationReceiver(
    std::unique_ptr<fjord::FjordChannel> channel)
    : channel_(std::move(channel)) {
    if (!channel_) {
        throw std::invalid_argument("FoamNordic observation receiver requires a channel.");
    }
}

std::optional<ObservationRecord> ObservationReceiver::receive(
    std::chrono::milliseconds timeout) {
    if (!channel_->wait_readable(timeout)) {
        return std::nullopt;
    }
    return read_frame(*channel_);
}

void ObservationReceiver::interrupt() noexcept { channel_->interrupt(); }

void ObservationReceiver::close() noexcept { channel_->close(); }

std::size_t LongshipObservation::byte_size() const noexcept {
    return sizeof(LongshipObservation) + source.size() + record.byte_size();
}

LongshipObservationStream::LongshipObservationStream(
    std::vector<ObservationSource> sources,
    ObservationRetention retention)
    : sources_(std::move(sources)), retention_(retention) {
    retention_.validate();
    if (sources_.empty()) {
        throw std::invalid_argument(
            "Longship observation stream requires at least one source.");
    }
    for (std::size_t index = 0; index < sources_.size(); ++index) {
        if (sources_[index].name.empty() || !sources_[index].receiver) {
            throw std::invalid_argument(
                "Longship observation source requires a name and receiver.");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (sources_[previous].name == sources_[index].name) {
                throw std::invalid_argument(
                    "Longship observation source name is duplicated: "
                    + sources_[index].name);
            }
        }
    }
    workers_.reserve(sources_.size());
    try {
        for (std::size_t index = 0; index < sources_.size(); ++index) {
            workers_.emplace_back([this, index] { run(index); });
        }
    } catch (...) {
        stopped_.store(true, std::memory_order_release);
        for (auto& source : sources_) {
            source.receiver->interrupt();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        for (auto& source : sources_) {
            source.receiver->close();
        }
        throw;
    }
}

LongshipObservationStream::~LongshipObservationStream() { stop(); }

std::optional<LongshipObservation> LongshipObservationStream::receive(
    std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!available_.wait_for(
            lock, timeout, [this] {
                return stopped_.load(std::memory_order_acquire)
                       || !records_.empty();
            })) {
        return std::nullopt;
    }
    if (records_.empty()) {
        return std::nullopt;
    }
    auto observation = std::move(records_.front());
    bytes_ -= observation.byte_size();
    records_.pop_front();
    return observation;
}

void LongshipObservationStream::stop() noexcept {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    for (auto& source : sources_) {
        source.receiver->interrupt();
    }
    available_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    for (auto& source : sources_) {
        source.receiver->close();
    }
}

bool LongshipObservationStream::healthy() const noexcept {
    return healthy_.load(std::memory_order_acquire);
}

std::string LongshipObservationStream::failure() const {
    std::scoped_lock lock(mutex_);
    return failure_;
}

std::uint64_t LongshipObservationStream::dropped_records() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
}

void LongshipObservationStream::run(std::size_t source_index) noexcept {
    auto& source = sources_[source_index];
    try {
        while (!stopped_.load(std::memory_order_acquire)) {
            auto record = source.receiver->receive(std::chrono::milliseconds(100));
            if (record) {
                retain({0, source.name, std::move(*record)});
            }
        }
    } catch (const std::exception& error) {
        if (!stopped_.load(std::memory_order_acquire)) {
            std::scoped_lock lock(mutex_);
            healthy_.store(false, std::memory_order_release);
            if (failure_.empty()) {
                failure_ = source.name + ": " + error.what();
            }
        }
    }
}

void LongshipObservationStream::retain(LongshipObservation observation) {
    std::unique_lock lock(mutex_);
    if (stopped_.load(std::memory_order_acquire)) {
        return;
    }
    const auto observation_bytes = observation.byte_size();
    if (observation_bytes > retention_.max_bytes) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (retention_.overflow == ObservationOverflow::drop_newest
        && (records_.size() >= retention_.max_records
            || bytes_ + observation_bytes > retention_.max_bytes)) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    while (!records_.empty()
           && (records_.size() >= retention_.max_records
               || bytes_ + observation_bytes > retention_.max_bytes)) {
        bytes_ -= records_.front().byte_size();
        records_.pop_front();
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    observation.stream_index = next_stream_index_++;
    bytes_ += observation_bytes;
    records_.push_back(std::move(observation));
    lock.unlock();
    available_.notify_one();
}

}  // namespace foamnordic::adapter
