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

#include <cstdint>
#include <memory>
#include <string>

#include "foamnordic/fjord/channel.hpp"

namespace foamnordic::fjord {

enum class SharedMemoryRole {
    initiator,
    responder,
};

[[nodiscard]] std::unique_ptr<FjordChannel> create_shared_memory_channel(
    const std::string& name,
    std::unique_ptr<FjordChannel> wake_channel,
    SharedMemoryRole role = SharedMemoryRole::initiator,
    std::uint64_t slots = 16,
    std::uint64_t slot_payload = 1024 * 1024);

[[nodiscard]] std::unique_ptr<FjordChannel> connect_shared_memory_channel(
    const std::string& name,
    std::unique_ptr<FjordChannel> wake_channel,
    SharedMemoryRole role = SharedMemoryRole::responder);

}  // namespace foamnordic::fjord
