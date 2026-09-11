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
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

using namespace foamnordic::inference;
static void queued(ExecutionContextPool& pool, unsigned count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (pool.snapshot().waiting_contexts != count) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Admission queue timed out.");
        std::this_thread::yield();
    }
}
static void queued(ExecutionCapacityPool& pool, unsigned count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (pool.snapshot().waiting != count) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Capacity queue timed out.");
        std::this_thread::yield();
    }
}
int main() {
    ExecutionContextPool pool({1, 64});
    auto held = pool.acquire(64);
    std::vector<unsigned> order;
    std::vector<std::future<void>> requests;
    for (unsigned i = 0; i < 8; ++i) {
        requests.push_back(std::async(std::launch::async, [&, i] {
            auto lease = pool.acquire(i % 2 ? 32 : 64);
            // A one-slot pool exclusively owns this vector while leased.
            order.push_back(i);
        }));
        queued(pool, i + 1);
    }
    if (pool.snapshot().peak_waiting_contexts != 8)
        throw std::runtime_error("Admission queue peak was not recorded.");
    held = {};
    for (auto& request : requests) request.get();
    for (unsigned i = 0; i < order.size(); ++i)
        if (order[i] != i) throw std::runtime_error("Admission was not FIFO.");
    held = pool.acquire(64);
    auto cancelled = std::async(std::launch::async, [&] {
        try { auto lease = pool.acquire(1); }
        catch (const std::runtime_error&) { return true; }
        return false;
    });
    queued(pool, 1);
    pool.cancel();
    if (!cancelled.get()) throw std::runtime_error("Cancellation did not release waiter.");
    held = {};
    if (pool.snapshot().admitted_bytes || pool.snapshot().waiting_contexts)
        throw std::runtime_error("Admission credit or waiter leaked.");

    ExecutionCapacityPool capacity(4, 1);
    auto occupied = capacity.acquire(3);
    auto wide_started = std::promise<void>();
    auto wide = std::async(std::launch::async, [&] {
        auto lease = capacity.acquire(4);
        wide_started.set_value();
    });
    queued(capacity, 1);
    auto narrow = std::async(std::launch::async, [&] {
        auto lease = capacity.acquire(1);
    });
    narrow.get();
    queued(capacity, 1);
    auto second_narrow = std::async(std::launch::async, [&] {
        auto lease = capacity.acquire(1);
    });
    queued(capacity, 2);
    if (wide_started.get_future().wait_for(std::chrono::milliseconds(10))
        != std::future_status::timeout)
        throw std::runtime_error("Wide request ran without sufficient capacity.");
    occupied = {};
    wide.get();
    second_narrow.get();
    const auto capacity_snapshot = capacity.snapshot();
    if (capacity_snapshot.active || capacity_snapshot.waiting
        || capacity_snapshot.peak_active != 4
        || capacity_snapshot.peak_waiting != 2)
        throw std::runtime_error("Shared execution capacity accounting is invalid.");

    ExecutionCapacityPool cancelled_capacity(1);
    auto cancelled_occupied = cancelled_capacity.acquire(1);
    auto cancelled_capacity_waiter = std::async(std::launch::async, [&] {
        try { auto lease = cancelled_capacity.acquire(1); }
        catch (const std::runtime_error&) { return true; }
        return false;
    });
    queued(cancelled_capacity, 1);
    cancelled_capacity.cancel();
    if (!cancelled_capacity_waiter.get())
        throw std::runtime_error("Capacity cancellation did not release waiter.");
    cancelled_occupied = {};
    if (cancelled_capacity.snapshot().active
        || cancelled_capacity.snapshot().waiting)
        throw std::runtime_error("Cancelled execution capacity leaked.");
}
