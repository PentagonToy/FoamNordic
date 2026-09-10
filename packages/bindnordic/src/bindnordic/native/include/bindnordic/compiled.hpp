// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * BindNordic
 */
// clang-format on

#pragma once
#include <filesystem>
#include "bindnordic/kernel.hpp"

namespace bindnordic {
class CompiledKernel final : public PackedModelKernel {
public:
    CompiledKernel(const std::filesystem::path& library, std::uint32_t threads);
    ~CompiledKernel() override;
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
    BackendCapabilities capabilities() const noexcept override { return {false, true, true}; }
    Tensor evaluate(TensorView input, std::uint64_t index,
                           double time, std::uint32_t rank = 0) override;
private:
    void* library_{};
    using Predict = int (*)(const double*, std::size_t, std::size_t,
                           double*, std::size_t, std::size_t);
    Predict predict_{};
    using SerialPredict = int (*)(const double*, std::size_t, std::size_t,
                                 double*, std::size_t);
    SerialPredict serial_{};
    std::size_t inputs_{}, outputs_{};
    std::uint32_t threads_;
};
}
