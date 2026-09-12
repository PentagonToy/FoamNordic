// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\\ /  \\|  V  | |  \\| |/__\\| _ \\ _\\| |/ _/
 * | _| \\/ | /\\ | \\_/ | | | ' | \\/ | v / v | | \\__
 * |_| \\__/|_||_|_| |_| |_|\\__|\\__/|_|_\\__/|_|\\__/
 *
 * FoamNordic
 */
// clang-format on

#include "foamnordic/backend/inference/residency.hpp"

namespace foamnordic::inference {
namespace {

ResidencyRoute route(
    bindnordic::MemorySpace memory,
    const ResidencySupport& support) noexcept {
    switch (memory) {
        case bindnordic::MemorySpace::host:
            return ResidencyRoute::direct;
        case bindnordic::MemorySpace::pinned_host:
            return support.pinned_host_staging ? ResidencyRoute::staging
                                               : ResidencyRoute::unsupported;
        case bindnordic::MemorySpace::device:
            return support.device_staging ? ResidencyRoute::staging
                                          : ResidencyRoute::unsupported;
        case bindnordic::MemorySpace::unified:
            return support.unified_memory ? ResidencyRoute::direct
                                          : ResidencyRoute::unsupported;
    }
    return ResidencyRoute::unsupported;
}

}  // namespace

bool ResidencyPlan::supported() const noexcept {
    return input != ResidencyRoute::unsupported
           && output != ResidencyRoute::unsupported && completion_supported;
}

bool ResidencyPlan::direct() const noexcept {
    return supported() && input == ResidencyRoute::direct
           && output == ResidencyRoute::direct && !wait_for_event;
}

ResidencyPlan plan_residency(
    const bindnordic::BackendCapabilities& capabilities,
    ResidencySupport support) noexcept {
    if (!capabilities.valid()) {
        return {};
    }
    const auto event = capabilities.completion
                       == bindnordic::CompletionMode::event;
    return {
        route(capabilities.input_memory, support),
        route(capabilities.output_memory, support),
        !event || support.event_completion,
        event && support.event_completion,
        capabilities.preferred_batch_rows,
        capabilities.alignment_bytes,
    };
}

}  // namespace foamnordic::inference
