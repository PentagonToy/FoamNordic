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

#include "foamnordic/runtime/longship_supervisor.hpp"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace foamnordic::native {
namespace {

struct Child {
    pid_t pid{-1};
};

[[nodiscard]] int normalized_status(int status) noexcept {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 255;
}

[[noreturn]] void execute_child(const LongshipCommand& command) {
    if (!command.output.empty()) {
        const auto descriptor = ::open(
            command.output.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0660);
        if (descriptor < 0) {
            _exit(126);
        }
        if (::dup2(descriptor, STDOUT_FILENO) < 0
            || ::dup2(descriptor, STDERR_FILENO) < 0) {
            _exit(126);
        }
        ::close(descriptor);
    }

    std::vector<char*> arguments;
    arguments.reserve(command.arguments.size() + 1);
    for (const auto& argument : command.arguments) {
        arguments.push_back(const_cast<char*>(argument.c_str()));
    }
    arguments.push_back(nullptr);
    ::execvp(arguments.front(), arguments.data());
    _exit(errno == ENOENT ? 127 : 126);
}

[[nodiscard]] Child start_child(const LongshipCommand& command) {
    const auto pid = ::fork();
    if (pid < 0) {
        throw std::system_error(
            errno, std::generic_category(), "Cannot start Longship component");
    }
    if (pid == 0) {
        static_cast<void>(::setpgid(0, 0));
        execute_child(command);
    }
    static_cast<void>(::setpgid(pid, pid));
    return {pid};
}

void signal_group(const Child& child, int signal) noexcept {
    if (child.pid > 0) {
        static_cast<void>(::kill(-child.pid, signal));
    }
}

class ChildState {
public:
    explicit ChildState(Child child) : child_(child) {}

    ChildState(const ChildState&) = delete;
    ChildState& operator=(const ChildState&) = delete;

    void watch() {
        watcher_ = std::thread([this] {
            int raw_status = 0;
            pid_t result = -1;
            do {
                result = ::waitpid(child_.pid, &raw_status, 0);
            } while (result < 0 && errno == EINTR);

            {
                std::lock_guard lock(mutex_);
                status_ = result == child_.pid
                              ? normalized_status(raw_status)
                              : 255;
            }
            changed_.notify_all();
        });
    }

    [[nodiscard]] bool finished() const {
        std::lock_guard lock(mutex_);
        return status_.has_value();
    }

    [[nodiscard]] int status() const {
        std::lock_guard lock(mutex_);
        return status_.value_or(255);
    }

    void wait_for_change(std::chrono::milliseconds duration) const {
        std::unique_lock lock(mutex_);
        changed_.wait_for(lock, duration);
    }

    void terminate(std::chrono::milliseconds grace) {
        if (finished()) {
            return;
        }
        signal_group(child_, SIGTERM);
        wait_for_change(grace);
        if (!finished()) {
            signal_group(child_, SIGKILL);
        }
    }

    void join() {
        if (watcher_.joinable()) {
            watcher_.join();
        }
    }

private:
    Child child_;
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    std::optional<int> status_;
    std::thread watcher_;
};

[[nodiscard]] bool all_ready(
    const std::vector<std::filesystem::path>& files) {
    for (const auto& file : files) {
        std::error_code error;
        const auto status = std::filesystem::status(file, error);
        if (error
            || (status.type() != std::filesystem::file_type::regular
                && status.type() != std::filesystem::file_type::socket)) {
            return false;
        }
    }
    return true;
}

void remove_stale_readiness(
    const std::vector<std::filesystem::path>& files) {
    for (const auto& file : files) {
        std::error_code error;
        static_cast<void>(std::filesystem::remove(file, error));
        if (error) {
            throw std::filesystem::filesystem_error(
                "Cannot remove stale Longship readiness file",
                file,
                error);
        }
    }
}

void remove_readiness_noexcept(
    const std::vector<std::filesystem::path>& files) noexcept {
    for (const auto& file : files) {
        std::error_code ignored;
        static_cast<void>(std::filesystem::remove(file, ignored));
    }
}

class ReadinessCleanup {
public:
    explicit ReadinessCleanup(
        const std::vector<std::filesystem::path>& files)
        : files_(files) {}

    ~ReadinessCleanup() { remove_readiness_noexcept(files_); }

    ReadinessCleanup(const ReadinessCleanup&) = delete;
    ReadinessCleanup& operator=(const ReadinessCleanup&) = delete;

private:
    const std::vector<std::filesystem::path>& files_;
};

[[nodiscard]] bool wait_until_ready(
    const LongshipLaunch& launch,
    const ChildState& host,
    const LongshipStop* stop) {
    const auto deadline = std::chrono::steady_clock::now()
                          + launch.readiness_timeout;
    while (!all_ready(launch.host_ready_files)) {
        if (stop != nullptr && stop->stop_requested()) {
            return false;
        }
        if (host.finished()) {
            throw std::runtime_error(
                "ModelHost exited before Longship readiness.");
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw std::runtime_error(
                "Longship timed out waiting for ModelHost readiness.");
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - now);
        host.wait_for_change(
            std::min(remaining, std::chrono::milliseconds(20)));
    }
    return true;
}

}  // namespace

void LongshipStop::request_stop() noexcept {
    requested_.store(true, std::memory_order_release);
}

bool LongshipStop::stop_requested() const noexcept {
    return requested_.load(std::memory_order_acquire);
}

void LongshipCommand::validate() const {
    if (arguments.empty() || arguments.front().empty()) {
        throw std::invalid_argument(
            "Longship command requires an executable.");
    }
}

void LongshipLaunch::validate() const {
    host.validate();
    solver.validate();
    if (host_ready_files.empty()) {
        throw std::invalid_argument(
            "Longship requires at least one ModelHost readiness file.");
    }
    if (readiness_timeout <= std::chrono::milliseconds::zero()
        || termination_grace <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument(
            "Longship lifecycle timeouts must be positive.");
    }
}

LongshipResult sail_longship(
    const LongshipLaunch& launch,
    const LongshipStop* stop) {
    launch.validate();
    remove_stale_readiness(launch.host_ready_files);
    const ReadinessCleanup readiness_cleanup(launch.host_ready_files);

    ChildState host(start_child(launch.host));
    host.watch();
    try {
        if (!wait_until_ready(launch, host, stop)) {
            host.terminate(launch.termination_grace);
            host.join();
            return {130, host.status(), false, true};
        }
    } catch (...) {
        host.terminate(launch.termination_grace);
        host.join();
        throw;
    }

    ChildState solver(start_child(launch.solver));
    solver.watch();
    while (!host.finished() && !solver.finished()) {
        if (stop != nullptr && stop->stop_requested()) {
            host.terminate(launch.termination_grace);
            solver.terminate(launch.termination_grace);
            host.join();
            solver.join();
            return {solver.status(), host.status(), false, true};
        }
        solver.wait_for_change(std::chrono::milliseconds(100));
    }

    bool host_failed_first =
        host.finished() && solver.finished() && host.status() != 0;
    if (host.finished() && !solver.finished()) {
        if (host.status() == 0) {
            solver.wait_for_change(launch.termination_grace);
        }
        if (!solver.finished()) {
            host_failed_first = true;
            solver.terminate(launch.termination_grace);
        }
    } else {
        // ModelHost is a resident service and is not expected to exit merely
        // because the solver has completed. Ask it to stop immediately; the
        // grace period bounds shutdown after SIGTERM instead of delaying the
        // signal itself.
        host.terminate(launch.termination_grace);
    }
    host.join();
    solver.join();

    return {
        solver.status(),
        host.status(),
        host_failed_first,
        false,
    };
}

}  // namespace foamnordic::native
