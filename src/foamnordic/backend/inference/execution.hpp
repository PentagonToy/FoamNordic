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
#include <mutex>
#include <vector>

#include "foamnordic/backend/inference/smedja.hpp"

namespace foamnordic::inference {

struct ExecutionResources {
    std::uint32_t contexts{1};
    std::uint64_t maximum_working_bytes{16ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t profile_every{0};

    void validate() const;
};

struct ExecutionSnapshot {
    std::uint64_t requests{0};
    std::uint32_t active_contexts{0};
    std::uint32_t peak_active_contexts{0};
    std::uint64_t admitted_bytes{0};
    std::uint64_t peak_admitted_bytes{0};
    std::uint64_t retained_workspace_bytes{0};
    std::uint64_t profile_samples{0};
    std::uint64_t packing_nanoseconds{0};
    std::uint64_t inference_nanoseconds{0};
    std::uint64_t output_nanoseconds{0};
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
    bool cancelled_{false};
};

}  // namespace foamnordic::inference
