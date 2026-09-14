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

#include <atomic>
#include <mutex>

#include "bindnordic/kernel.hpp"
#include "foamnordic/backend/inference/artifact.hpp"
#include "foamnordic/backend/inference/execution.hpp"
#include "foamnordic/backend/inference/smedja.hpp"
#include "foamnordic/backend/inference/runner.hpp"

namespace foamnordic::inference {

using bindnordic::BackendCapabilities;
using bindnordic::PackedModelKernel;

class ArtifactModelKernel final : public ModelKernel {
public:
    ArtifactModelKernel(
        ModelArtifact artifact,
        PackedModelKernel& kernel,
        ExecutionResources resources = {});

    [[nodiscard]] TensorMap evaluate(
        const TensorMap& inputs,
        const std::vector<std::uint64_t>& active_cells,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank = 0) override;

    [[nodiscard]] bool owns_backend_synchronization() const noexcept override;
    [[nodiscard]] ExecutionSnapshot execution_snapshot() const override;
    [[nodiscard]] bool telemetry_enabled() const noexcept override;
    void observe_exchange(
        std::uint64_t rows,
        std::uint64_t active_cells,
        std::uint64_t input_payload_bytes,
        std::uint64_t output_payload_bytes) noexcept override;
    void cancel_pending() noexcept override;

    [[nodiscard]] const ModelArtifact& artifact() const noexcept;

private:
    ModelArtifact artifact_;
    PackedModelKernel& kernel_;
    Smedja smedja_;
    ExecutionContextPool contexts_;
    BackendCapabilities capabilities_;
    std::shared_ptr<ExecutionCapacityPool> capacity_;
    std::uint32_t capacity_units_{1};
    ExecutionContextPool backend_gate_{ExecutionResources{1}};
    std::uint64_t profile_every_{0};
    std::atomic<std::uint64_t> profile_samples_{0};
    std::atomic<std::uint64_t> packing_nanoseconds_{0};
    std::atomic<std::uint64_t> inference_nanoseconds_{0};
    std::atomic<std::uint64_t> output_nanoseconds_{0};
    std::atomic<std::uint64_t> admission_nanoseconds_{0};
    std::atomic<std::uint64_t> backend_wait_nanoseconds_{0};
    std::atomic<std::uint64_t> completed_exchanges_{0};
    std::atomic<std::uint64_t> rows_{0};
    std::atomic<std::uint64_t> active_cells_{0};
    std::atomic<std::uint64_t> input_payload_bytes_{0};
    std::atomic<std::uint64_t> output_payload_bytes_{0};
    std::atomic<std::uint64_t> peak_rows_{0};
    std::atomic<std::uint64_t> peak_active_cells_{0};
    std::atomic<std::uint64_t> peak_input_payload_bytes_{0};
    std::atomic<std::uint64_t> peak_output_payload_bytes_{0};
};

}  // namespace foamnordic::inference
