#include "foamnordic/runtime/executor.hpp"
#include <array>
#include <atomic>
#include <stdexcept>

int main() {
    FnoExecutor executor;
    std::array<std::thread::id, 4> first;
    executor.run(4, [&](auto i) { first[i] = std::this_thread::get_id(); });
    for (unsigned call = 0; call < 100; ++call) {
        executor.run(4, [&](auto i) {
            if (first[i] != std::this_thread::get_id())
                throw std::runtime_error("Executor replaced a warm worker.");
        });
        executor.run(1, [](auto) {});
    }
    bool rejected = false;
    try { executor.run(4, [](auto i) { if (i == 1) throw std::runtime_error("test"); }); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Worker exception was lost.");
    executor.run(4, [](auto) {});
    rejected = false;
    try { executor.run(0, [](auto) {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Zero workers accepted.");
    std::atomic<unsigned> total{0};
    auto concurrent = [&] {
        for (unsigned call = 0; call < 100; ++call)
            executor.run(3, [&](auto) { ++total; });
    };
    std::thread one(concurrent), two(concurrent);
    one.join(); two.join();
    if (total != 600) throw std::runtime_error("Concurrent submission was lost.");
}
