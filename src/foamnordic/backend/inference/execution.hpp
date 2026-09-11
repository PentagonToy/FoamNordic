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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "foamnordic/backend/inference/smedja.hpp"

namespace foamnordic::inference {

class ExecutionCapacityPool;

struct ExecutionResources {
    std::uint32_t contexts{1};
    std::uint64_t maximum_working_bytes{16ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t profile_every{0};
    std::shared_ptr<ExecutionCapacityPool> capacity;
    std::uint32_t capacity_units{1};

    void validate() const;
};

struct ExecutionSnapshot {
    std::uint64_t requests{0};
    std::uint64_t completed_exchanges{0};
    std::uint64_t rows{0};
    std::uint64_t active_cells{0};
    std::uint64_t input_payload_bytes{0};
    std::uint64_t output_payload_bytes{0};
    std::uint64_t peak_rows{0};
    std::uint64_t peak_active_cells{0};
    std::uint64_t peak_input_payload_bytes{0};
    std::uint64_t peak_output_payload_bytes{0};
    std::uint32_t active_contexts{0};
    std::uint32_t peak_active_contexts{0};
    std::uint64_t admitted_bytes{0};
    std::uint64_t peak_admitted_bytes{0};
    std::uint64_t retained_workspace_bytes{0};
    std::uint64_t profile_samples{0};
    std::uint64_t packing_nanoseconds{0};
    std::uint64_t inference_nanoseconds{0};
    std::uint64_t output_nanoseconds{0};
    std::uint64_t admission_nanoseconds{0};
    std::uint64_t backend_wait_nanoseconds{0};
    std::uint64_t waiting_contexts{0};
    std::uint64_t peak_waiting_contexts{0};
};

struct ReceiveCreditSnapshot {
    std::uint64_t requests{0};
    std::uint64_t active_bytes{0};
    std::uint64_t peak_active_bytes{0};
};

struct ExecutionCapacitySnapshot {
    std::uint32_t capacity{0};
    std::uint32_t active{0};
    std::uint32_t peak_active{0};
    std::uint64_t waiting{0};
    std::uint64_t peak_waiting{0};
};

class ExecutionCapacityPool final {
public:
    class Lease final {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

    private:
        friend class ExecutionCapacityPool;
        Lease(ExecutionCapacityPool& owner, std::uint32_t units) noexcept;
        void release() noexcept;

        ExecutionCapacityPool* owner_{nullptr};
        std::uint32_t units_{0};
    };

    explicit ExecutionCapacityPool(
        std::uint32_t capacity,
        std::uint32_t maximum_backfills = 8);

    [[nodiscard]] Lease acquire(std::uint32_t units);
    void cancel() noexcept;
    [[nodiscard]] ExecutionCapacitySnapshot snapshot() const;

private:
    struct Waiter {
        std::uint32_t units;
        std::uint32_t bypasses{0};
    };

    void release(std::uint32_t units) noexcept;

    std::uint32_t capacity_;
    std::uint32_t maximum_backfills_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::uint32_t active_{0};
    std::uint32_t peak_active_{0};
    std::uint64_t peak_waiting_{0};
    bool cancelled_{false};
    std::deque<Waiter*> waiters_;
};

class ReceiveCreditPool final {
public:
    class Reservation final {
    public:
        explicit Reservation(ReceiveCreditPool& owner) noexcept;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        Reservation(Reservation&&) = delete;
        Reservation& operator=(Reservation&&) = delete;
        ~Reservation();

        void acquire(std::uint64_t bytes);
        void seal() noexcept;

    private:
        ReceiveCreditPool* owner_;
        std::uint64_t bytes_{0};
        bool receiving_{false};
    };

    explicit ReceiveCreditPool(std::uint64_t maximum_bytes);

    void acquire(std::uint64_t bytes);
    void release(std::uint64_t bytes) noexcept;
    void cancel() noexcept;
    [[nodiscard]] ReceiveCreditSnapshot snapshot() const;

private:
    std::uint64_t maximum_bytes_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::uint64_t requests_{0};
    std::uint64_t active_bytes_{0};
    std::uint64_t peak_active_bytes_{0};
    bool cancelled_{false};
    Reservation* receiver_{nullptr};
    std::deque<Reservation*> receivers_;
};

class ExecutionContextPool final {
public:
    class Lease final {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

        [[nodiscard]] SmedjaWorkspace& workspace();

    private:
        friend class ExecutionContextPool;
        Lease(
            ExecutionContextPool& owner,
            std::size_t index,
            std::uint64_t admitted_bytes) noexcept;
        void release() noexcept;

        ExecutionContextPool* owner_{nullptr};
        std::size_t index_{0};
        std::uint64_t admitted_bytes_{0};
    };

    explicit ExecutionContextPool(ExecutionResources resources);

    [[nodiscard]] Lease acquire(std::uint64_t working_bytes);
    void cancel() noexcept;
    [[nodiscard]] ExecutionSnapshot snapshot() const;

private:
    struct Slot {
        SmedjaWorkspace workspace;
        bool active{false};
        std::uint64_t retained_bytes{0};
    };

    void release(std::size_t index, std::uint64_t admitted_bytes) noexcept;
    [[nodiscard]] std::uint64_t retained_bytes_locked() const noexcept;

    ExecutionResources resources_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Slot> slots_;
    std::uint32_t active_contexts_{0};
    std::uint32_t peak_active_contexts_{0};
    std::uint64_t requests_{0};
    std::uint64_t admitted_bytes_{0};
    std::uint64_t peak_admitted_bytes_{0};
    std::uint64_t peak_waiting_contexts_{0};
    bool cancelled_{false};
    std::deque<const void*> waiters_;
};

}  // namespace foamnordic::inference
