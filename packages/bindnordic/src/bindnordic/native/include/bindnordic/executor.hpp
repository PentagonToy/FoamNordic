#pragma once

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace bindnordic {

class Executor {
    std::mutex dispatch_, mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> threads_;
    std::function<void(std::size_t)> task_;
    std::exception_ptr failure_;
    std::size_t generation_{0}, pending_{0}, active_{0};
    bool stopping_{false};

public:
    ~Executor() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    void run(std::size_t count, const std::function<void(std::size_t)>& task) {
        if (!count) throw std::invalid_argument("Executor worker count must be positive.");
        std::lock_guard<std::mutex> dispatch(dispatch_);
        std::unique_lock<std::mutex> lock(mutex_);
        while (threads_.size() + 1 < count) {
            const auto index = threads_.size();
            const auto initial = generation_;
            threads_.emplace_back([this, index, initial] {
                auto seen = initial;
                std::unique_lock<std::mutex> guard(mutex_);
                for (;;) {
                    ready_.wait(guard, [&] { return stopping_ || generation_ != seen; });
                    if (stopping_) return;
                    seen = generation_;
                    if (index >= active_) continue;
                    guard.unlock();
                    try { task_(index); }
                    catch (...) {
                        std::lock_guard<std::mutex> failure_lock(mutex_);
                        if (!failure_) failure_ = std::current_exception();
                    }
                    guard.lock();
                    if (--pending_ == 0) done_.notify_one();
                }
            });
        }
        task_ = task;
        failure_ = nullptr;
        active_ = pending_ = count - 1;
        ++generation_;
        lock.unlock();
        ready_.notify_all();
        try { task(count - 1); }
        catch (...) {
            std::lock_guard<std::mutex> failure_lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
        lock.lock();
        done_.wait(lock, [&] { return pending_ == 0; });
        task_ = {};
        if (failure_) std::rethrow_exception(failure_);
    }
};

template<class Function>
void parallel(std::size_t items, std::size_t workers, Function function) {
    if (!items) return;
    workers = std::min(items, std::max<std::size_t>(1, workers));
    if (workers == 1) { function(0, 0, items); return; }
    static Executor executor;
    executor.run(workers, [&](std::size_t worker) {
        const auto quotient = items / workers;
        const auto remainder = items % workers;
        const auto first = worker * quotient + std::min(worker, remainder);
        function(worker, first, first + quotient + (worker < remainder));
    });
}

}  // namespace bindnordic
