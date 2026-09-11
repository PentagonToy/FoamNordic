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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "bindnordic/kernel.hpp"

namespace bindnordic {

struct OnnxOptions {
    std::int32_t intra_op_threads{1};
    std::int32_t inter_op_threads{1};

    void validate() const;
};

[[nodiscard]] BackendCapabilities onnx_backend_capabilities() noexcept;

class OnnxPackedKernel final : public PackedModelKernel {
public:
    explicit OnnxPackedKernel(
        const std::filesystem::path& model_path,
        OnnxOptions options = {});
    explicit OnnxPackedKernel(
        std::vector<std::byte> model_bytes,
        OnnxOptions options = {});
    ~OnnxPackedKernel() override;

    OnnxPackedKernel(const OnnxPackedKernel&) = delete;
    OnnxPackedKernel& operator=(const OnnxPackedKernel&) = delete;
    OnnxPackedKernel(OnnxPackedKernel&&) noexcept;
    OnnxPackedKernel& operator=(OnnxPackedKernel&&) noexcept;

    [[nodiscard]] BackendCapabilities capabilities() const noexcept override;

    [[nodiscard]] Tensor evaluate(
        TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) override;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace bindnordic
