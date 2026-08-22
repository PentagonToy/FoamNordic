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
constexpr std::uint16_t version = 1;
constexpr std::size_t header_size = 32;
constexpr std::size_t field_size = 40;
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

void ObservationReceiver::close() noexcept { channel_->close(); }

}  // namespace foamnordic::adapter
