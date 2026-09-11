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

ExecutionCapacityPool::Lease::Lease(
    ExecutionCapacityPool& owner,
    std::uint32_t units) noexcept
    : owner_(&owner), units_(units) {}

ExecutionCapacityPool::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_), units_(other.units_) {
    other.owner_ = nullptr;
}

ExecutionCapacityPool::Lease& ExecutionCapacityPool::Lease::operator=(
    Lease&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        units_ = other.units_;
        other.owner_ = nullptr;
    }
    return *this;
}

ExecutionCapacityPool::Lease::~Lease() { release(); }

void ExecutionCapacityPool::Lease::release() noexcept {
    if (owner_ != nullptr) {
        owner_->release(units_);
        owner_ = nullptr;
    }
}

ExecutionCapacityPool::ExecutionCapacityPool(
    std::uint32_t capacity,
    std::uint32_t maximum_backfills)
    : capacity_(capacity), maximum_backfills_(maximum_backfills) {
    if (capacity_ == 0) {
        throw std::invalid_argument("Execution capacity must be positive.");
    }
}

ExecutionCapacityPool::Lease ExecutionCapacityPool::acquire(
    std::uint32_t units) {
    if (units == 0 || units > capacity_) {
        throw std::invalid_argument(
            "Execution request must fit the shared capacity.");
    }
    std::unique_lock lock(mutex_);
    Waiter waiter{units};
    waiters_.push_back(&waiter);
    peak_waiting_ = std::max<std::uint64_t>(peak_waiting_, waiters_.size());
    available_.wait(lock, [&] {
        if (cancelled_) {
            return true;
        }
        const auto free = capacity_ - active_;
        if (waiter.units > free) {
            return false;
        }
        for (const auto* earlier : waiters_) {
            if (earlier == &waiter) {
                return true;
            }
            if (earlier->units <= free
                || earlier->bypasses >= maximum_backfills_) {
                return false;
            }
        }
        return false;
    });
    if (cancelled_) {
        std::erase(waiters_, &waiter);
        throw std::runtime_error("Shared execution was cancelled.");
    }
    const auto free = capacity_ - active_;
    for (auto* earlier : waiters_) {
        if (earlier == &waiter) {
            break;
        }
        if (earlier->units > free) {
            ++earlier->bypasses;
        }
    }
    std::erase(waiters_, &waiter);
    active_ += units;
    peak_active_ = std::max(peak_active_, active_);
    available_.notify_all();
    return Lease(*this, units);
}

void ExecutionCapacityPool::release(std::uint32_t units) noexcept {
    {
        std::scoped_lock lock(mutex_);
        active_ -= units;
    }
    available_.notify_all();
}

void ExecutionCapacityPool::cancel() noexcept {
    {
        std::scoped_lock lock(mutex_);
        cancelled_ = true;
    }
    available_.notify_all();
}

ExecutionCapacitySnapshot ExecutionCapacityPool::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {
        capacity_, active_, peak_active_, waiters_.size(), peak_waiting_};
}

void ExecutionResources::validate() const {
    if (contexts == 0) {
        throw std::invalid_argument("Execution context count must be positive.");
    }
    if (maximum_working_bytes == 0) {
        throw std::invalid_argument("Execution working-memory limit must be positive.");
    }
    if (capacity_units == 0) {
        throw std::invalid_argument("Execution capacity units must be positive.");
    }
}

ReceiveCreditPool::Reservation::Reservation(ReceiveCreditPool& owner) noexcept
    : owner_(&owner) {}

ReceiveCreditPool::Reservation::~Reservation() {
    owner_->release(bytes_);
    seal();
}

void ReceiveCreditPool::Reservation::acquire(std::uint64_t bytes) {
    if (bytes > owner_->maximum_bytes_ - bytes_) {
        throw std::length_error("Model exchange exceeds the shared receive-memory limit.");
    }
    if (!receiving_) {
        std::unique_lock lock(owner_->mutex_);
        owner_->receivers_.push_back(this);
        owner_->available_.wait(lock, [&] {
            return owner_->cancelled_ || (owner_->receiver_ == nullptr
                && owner_->receivers_.front() == this);
        });
        if (owner_->cancelled_) {
            std::erase(owner_->receivers_, this);
            throw std::runtime_error("Model input reception was cancelled.");
        }
        owner_->receivers_.pop_front();
        owner_->receiver_ = this;
        receiving_ = true;
    }
    owner_->acquire(bytes);
    bytes_ += bytes;
}

void ReceiveCreditPool::Reservation::seal() noexcept {
    if (receiving_) {
        {
            std::scoped_lock lock(owner_->mutex_);
            owner_->receiver_ = nullptr;
            receiving_ = false;
        }
        owner_->available_.notify_all();
    }
}

ReceiveCreditPool::ReceiveCreditPool(std::uint64_t maximum_bytes)
    : maximum_bytes_(maximum_bytes) {
    if (maximum_bytes_ == 0) {
        throw std::invalid_argument("Model receive-memory limit must be positive.");
    }
}

void ReceiveCreditPool::acquire(std::uint64_t bytes) {
    if (bytes > maximum_bytes_) {
        throw std::length_error(
            "Model tensor exceeds the shared receive-memory limit: required "
            + std::to_string(bytes) + " bytes, limit "
            + std::to_string(maximum_bytes_) + " bytes.");
    }
    std::unique_lock lock(mutex_);
    available_.wait(lock, [&] {
        return cancelled_ || bytes <= maximum_bytes_ - active_bytes_;
    });
    if (cancelled_) {
        throw std::runtime_error("Model input reception was cancelled.");
    }
    ++requests_;
    active_bytes_ += bytes;
    peak_active_bytes_ = std::max(peak_active_bytes_, active_bytes_);
}

void ReceiveCreditPool::release(std::uint64_t bytes) noexcept {
    {
        std::scoped_lock lock(mutex_);
        active_bytes_ -= bytes;
    }
    available_.notify_all();
}

void ReceiveCreditPool::cancel() noexcept {
    {
        std::scoped_lock lock(mutex_);
        cancelled_ = true;
    }
    available_.notify_all();
}

ReceiveCreditSnapshot ReceiveCreditPool::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {requests_, active_bytes_, peak_active_bytes_};
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
    const char ticket{};
    waiters_.push_back(&ticket);
    available_.wait(lock, [&] {
        if (cancelled_) {
            return true;
        }
        if (waiters_.front() != &ticket) {
            peak_waiting_contexts_ = std::max<std::uint64_t>(
                peak_waiting_contexts_, waiters_.size());
            return false;
        }
        auto can_admit = [&] {
            const auto retained = retained_bytes_locked();
            return std::any_of(slots_.begin(), slots_.end(), [&](const auto& slot) {
                if (slot.active) {
                    return false;
                }
                // Reusing a slot transfers ownership of its retained workspace to
                // the active request.  Count every other idle workspace, but do
                // not count this slot twice as both retained and active memory.
                const auto idle_elsewhere = retained - slot.retained_bytes;
                if (idle_elsewhere > resources_.maximum_working_bytes
                    || admitted_bytes_ > resources_.maximum_working_bytes - idle_elsewhere) {
                    return false;
                }
                return working_bytes <= resources_.maximum_working_bytes
                    - idle_elsewhere - admitted_bytes_;
            });
        };
        if (can_admit()) {
            return true;
        }
        // Idle retained capacity is a cache, never an admission dependency.
        // Drop it when it alone would otherwise prevent queued work.
        for (auto& slot : slots_) {
            if (slot.active) {
                continue;
            }
            if (slot.retained_bytes != 0) {
                slot.workspace.release();
                slot.retained_bytes = 0;
                if (can_admit()) {
                    return true;
                }
            }
        }
        peak_waiting_contexts_ = std::max<std::uint64_t>(
            peak_waiting_contexts_, waiters_.size());
        return false;
    });
    if (cancelled_) {
        std::erase(waiters_, &ticket);
        throw std::runtime_error("Model execution was cancelled.");
    }
    waiters_.pop_front();
    const auto retained = retained_bytes_locked();
    const auto slot = std::find_if(
        slots_.begin(), slots_.end(), [&](const auto& candidate) {
            if (candidate.active) {
                return false;
            }
            const auto idle_elsewhere = retained - candidate.retained_bytes;
            return idle_elsewhere <= resources_.maximum_working_bytes
                && admitted_bytes_ <= resources_.maximum_working_bytes - idle_elsewhere
                && working_bytes <= resources_.maximum_working_bytes
                    - idle_elsewhere - admitted_bytes_;
        });
    const auto index = static_cast<std::size_t>(slot - slots_.begin());
    ++requests_;
    slot->active = true;
    // From this point the capacity belongs to the active lease and is covered
    // by admitted_bytes_.  It becomes retained ownership again on release.
    slot->retained_bytes = 0;
    ++active_contexts_;
    admitted_bytes_ += working_bytes;
    peak_active_contexts_ = std::max(peak_active_contexts_, active_contexts_);
    peak_admitted_bytes_ = std::max(peak_admitted_bytes_, admitted_bytes_);
    available_.notify_all();
    return Lease(*this, index, working_bytes);
}

void ExecutionContextPool::cancel() noexcept {
    {
        std::scoped_lock lock(mutex_);
        cancelled_ = true;
    }
    available_.notify_all();
}

ExecutionSnapshot ExecutionContextPool::snapshot() const {
    std::scoped_lock lock(mutex_);
    ExecutionSnapshot result;
    result.requests = requests_;
    result.active_contexts = active_contexts_;
    result.peak_active_contexts = peak_active_contexts_;
    result.admitted_bytes = admitted_bytes_;
    result.peak_admitted_bytes = peak_admitted_bytes_;
    result.retained_workspace_bytes = retained_bytes_locked();
    result.waiting_contexts = waiters_.size();
    result.peak_waiting_contexts = peak_waiting_contexts_;
    return result;
}

std::uint64_t ExecutionContextPool::retained_bytes_locked() const noexcept {
    std::uint64_t total = 0;
    for (const auto& slot : slots_) {
        const auto retained = slot.retained_bytes;
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
        slot.retained_bytes = slot.workspace.retained_bytes();
        slot.active = false;
        --active_contexts_;
        admitted_bytes_ -= admitted_bytes;
        auto retained = retained_bytes_locked();
        if (retained > resources_.maximum_working_bytes) {
            slot.workspace.release();
            slot.retained_bytes = 0;
            retained = retained_bytes_locked();
        }
        for (auto& candidate : slots_) {
            if (retained <= resources_.maximum_working_bytes) {
                break;
            }
            if (!candidate.active && &candidate != &slot) {
                candidate.workspace.release();
                candidate.retained_bytes = 0;
                retained = retained_bytes_locked();
            }
        }
    }
    available_.notify_all();
}

}  // namespace foamnordic::inference
