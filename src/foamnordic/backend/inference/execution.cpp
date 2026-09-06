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

#include "foamnordic/backend/inference/execution.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace foamnordic::inference {

void ExecutionResources::validate() const {
    if (contexts == 0) {
        throw std::invalid_argument("Execution context count must be positive.");
    }
    if (maximum_working_bytes == 0) {
        throw std::invalid_argument("Execution working-memory limit must be positive.");
    }
}

ExecutionContextPool::Lease::Lease(
    ExecutionContextPool& owner,
    std::size_t index,
    std::uint64_t admitted_bytes) noexcept
    : owner_(&owner), index_(index), admitted_bytes_(admitted_bytes) {}

ExecutionContextPool::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_),
      index_(other.index_),
      admitted_bytes_(other.admitted_bytes_) {
    other.owner_ = nullptr;
}

ExecutionContextPool::Lease& ExecutionContextPool::Lease::operator=(
    Lease&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        index_ = other.index_;
        admitted_bytes_ = other.admitted_bytes_;
        other.owner_ = nullptr;
    }
    return *this;
}

ExecutionContextPool::Lease::~Lease() { release(); }

SmedjaWorkspace& ExecutionContextPool::Lease::workspace() {
    if (owner_ == nullptr) {
        throw std::logic_error("Execution context lease is not active.");
    }
    return owner_->slots_[index_].workspace;
}

void ExecutionContextPool::Lease::release() noexcept {
    if (owner_ != nullptr) {
        owner_->release(index_, admitted_bytes_);
        owner_ = nullptr;
    }
}

ExecutionContextPool::ExecutionContextPool(ExecutionResources resources)
    : resources_(resources), slots_(resources.contexts) {
    resources_.validate();
}

ExecutionContextPool::Lease ExecutionContextPool::acquire(
    std::uint64_t working_bytes) {
    if (working_bytes > resources_.maximum_working_bytes) {
        throw std::length_error(
            "Model request exceeds the execution working-memory limit: required "
            + std::to_string(working_bytes) + " bytes, limit "
            + std::to_string(resources_.maximum_working_bytes) + " bytes.");
    }
    std::unique_lock lock(mutex_);
    available_.wait(lock, [&] {
        const auto has_context = std::any_of(
            slots_.begin(), slots_.end(), [](const auto& slot) {
                return !slot.active;
            });
        return has_context
            && working_bytes <= resources_.maximum_working_bytes - admitted_bytes_;
    });
    const auto slot = std::find_if(
        slots_.begin(), slots_.end(), [](const auto& candidate) {
            return !candidate.active;
        });
    const auto index = static_cast<std::size_t>(slot - slots_.begin());
    ++requests_;
    slot->active = true;
    ++active_contexts_;
    admitted_bytes_ += working_bytes;
    peak_active_contexts_ = std::max(peak_active_contexts_, active_contexts_);
    peak_admitted_bytes_ = std::max(peak_admitted_bytes_, admitted_bytes_);
    return Lease(*this, index, working_bytes);
}

ExecutionSnapshot ExecutionContextPool::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {
        requests_,
        active_contexts_,
        peak_active_contexts_,
        admitted_bytes_,
        peak_admitted_bytes_,
        retained_bytes_locked(),
    };
}

std::uint64_t ExecutionContextPool::retained_bytes_locked() const noexcept {
    std::uint64_t total = 0;
    for (const auto& slot : slots_) {
        const auto retained = static_cast<std::uint64_t>(
            slot.workspace.retained_bytes());
        if (retained > std::numeric_limits<std::uint64_t>::max() - total) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        total += retained;
    }
    return total;
}

void ExecutionContextPool::release(
    std::size_t index,
    std::uint64_t admitted_bytes) noexcept {
    {
        std::scoped_lock lock(mutex_);
        auto& slot = slots_[index];
        slot.active = false;
        --active_contexts_;
        admitted_bytes_ -= admitted_bytes;
        auto retained = retained_bytes_locked();
        if (retained > resources_.maximum_working_bytes) {
            slot.workspace.release();
            retained = retained_bytes_locked();
        }
        for (auto& candidate : slots_) {
            if (retained <= resources_.maximum_working_bytes) {
                break;
            }
            if (!candidate.active && &candidate != &slot) {
                candidate.workspace.release();
                retained = retained_bytes_locked();
            }
        }
    }
    available_.notify_all();
}

}  // namespace foamnordic::inference
