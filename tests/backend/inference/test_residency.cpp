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

#include <cassert>

#include "foamnordic/backend/inference/residency.hpp"

namespace {

using foamnordic::inference::ResidencyRoute;
using foamnordic::inference::ResidencySupport;
using foamnordic::inference::plan_residency;

void test_host_is_direct() {
    bindnordic::BackendCapabilities capabilities;
    capabilities.preferred_batch_rows = 128;
    capabilities.alignment_bytes = 64;
    const auto plan = plan_residency(capabilities);
    assert(plan.supported());
    assert(plan.direct());
    assert(plan.preferred_batch_rows == 128);
    assert(plan.alignment_bytes == 64);
}

void test_staging_is_explicit() {
    bindnordic::BackendCapabilities capabilities;
    capabilities.input_memory = bindnordic::MemorySpace::device;
    capabilities.output_memory = bindnordic::MemorySpace::pinned_host;
    assert(!plan_residency(capabilities).supported());

    ResidencySupport support;
    support.device_staging = true;
    support.pinned_host_staging = true;
    const auto plan = plan_residency(capabilities, support);
    assert(plan.supported());
    assert(!plan.direct());
    assert(plan.input == ResidencyRoute::staging);
    assert(plan.output == ResidencyRoute::staging);
}

void test_events_require_support() {
    bindnordic::BackendCapabilities capabilities;
    capabilities.completion = bindnordic::CompletionMode::event;
    assert(!plan_residency(capabilities).direct());

    ResidencySupport support;
    support.event_completion = true;
    const auto plan = plan_residency(capabilities, support);
    assert(plan.supported());
    assert(plan.wait_for_event);
    assert(!plan.direct());
}

void test_unified_memory_is_opt_in() {
    bindnordic::BackendCapabilities capabilities;
    capabilities.input_memory = bindnordic::MemorySpace::unified;
    capabilities.output_memory = bindnordic::MemorySpace::unified;
    assert(!plan_residency(capabilities).supported());

    ResidencySupport support;
    support.unified_memory = true;
    assert(plan_residency(capabilities, support).direct());
}

}  // namespace

int main() {
    test_host_is_direct();
    test_staging_is_explicit();
    test_events_require_support();
    test_unified_memory_is_opt_in();
}
