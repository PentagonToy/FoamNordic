#include "foamnordic/backend/connectors/plugin.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "bindnordic/onnx.hpp"

namespace {

void set_error(char* destination, std::size_t capacity, const char* message) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    const auto size = std::min(capacity - 1, std::strlen(message));
    std::memcpy(destination, message, size);
    destination[size] = '\0';
}

void* create(
    const void* model,
    std::uint64_t model_bytes,
    std::uint32_t threads,
    char* error,
    std::size_t error_capacity) {
    try {
        if (model == nullptr || model_bytes == 0 || threads == 0) {
            throw std::invalid_argument("ONNX connector received invalid model data.");
        }
        const auto* first = static_cast<const std::byte*>(model);
        return new bindnordic::OnnxPackedKernel(
            std::vector<std::byte>(first, first + model_bytes),
            {static_cast<std::int32_t>(threads), 1});
    } catch (const std::exception& exception) {
        set_error(error, error_capacity, exception.what());
        return nullptr;
    }
}

void destroy(void* model) {
    delete static_cast<bindnordic::OnnxPackedKernel*>(model);
}

int evaluate(
    void* model,
    const void* input,
    std::uint64_t input_bytes,
    std::uint64_t rows,
    std::uint64_t columns,
    std::uint8_t element,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t rank,
    foamnordic_connector_result_v1* result,
    char* error,
    std::size_t error_capacity) {
    try {
        if (model == nullptr || input == nullptr || result == nullptr) {
            throw std::invalid_argument("ONNX connector received an invalid evaluation request.");
        }
        const auto* first = static_cast<const std::byte*>(input);
        const bindnordic::TensorView features{
            "input",
            static_cast<bindnordic::Element>(element),
            {rows, columns},
            std::span<const std::byte>(first, input_bytes),
        };
        auto output = static_cast<bindnordic::OnnxPackedKernel*>(model)->evaluate(
            features, exchange_index, physical_time, rank);
        if (output.shape.size() != 2) {
            throw std::runtime_error("ONNX connector produced a tensor that is not rank two.");
        }
        auto owner = std::make_unique<bindnordic::Tensor>(std::move(output));
        result->data = owner->bytes.data();
        result->bytes = owner->bytes.size();
        result->rows = owner->shape[0];
        result->columns = owner->shape[1];
        result->element = static_cast<std::uint8_t>(owner->element);
        result->owner = owner.release();
        return 0;
    } catch (const std::exception& exception) {
        set_error(error, error_capacity, exception.what());
        return 1;
    }
}

void release(void* owner) {
    delete static_cast<bindnordic::Tensor*>(owner);
}

const auto capabilities = bindnordic::onnx_backend_capabilities();

const foamnordic_connector_api_v1 api{
    FOAMNORDIC_CONNECTOR_ABI_V1,
    "onnx",
    static_cast<std::uint8_t>(capabilities.concurrent_evaluation),
    static_cast<std::uint8_t>(capabilities.row_independent),
    static_cast<std::uint8_t>(capabilities.owns_compute_threads),
    create,
    destroy,
    evaluate,
    release,
};

const foamnordic_connector_memory_api_v1 memory_api{
    FOAMNORDIC_CONNECTOR_MEMORY_ABI_V1,
    FOAMNORDIC_MEMORY_HOST_V1,
    FOAMNORDIC_MEMORY_HOST_V1,
    static_cast<std::uint8_t>(capabilities.persistent_device_state),
    FOAMNORDIC_COMPLETION_SYNCHRONOUS_V1,
    static_cast<std::uint8_t>(capabilities.workspace_recreated_on_shape_change),
    capabilities.preferred_batch_rows,
    capabilities.alignment_bytes,
};

}  // namespace

extern "C" const foamnordic_connector_api_v1* foamnordic_connector_v1() {
    return &api;
}

extern "C" const foamnordic_connector_memory_api_v1*
foamnordic_connector_memory_v1() {
    return &memory_api;
}
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
