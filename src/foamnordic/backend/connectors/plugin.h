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

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FOAMNORDIC_CONNECTOR_ABI_V1 1u
#define FOAMNORDIC_CONNECTOR_MEMORY_ABI_V1 1u

typedef enum foamnordic_connector_memory_space_v1 {
    FOAMNORDIC_MEMORY_HOST_V1 = 0,
    FOAMNORDIC_MEMORY_PINNED_HOST_V1 = 1,
    FOAMNORDIC_MEMORY_DEVICE_V1 = 2,
    FOAMNORDIC_MEMORY_UNIFIED_V1 = 3,
} foamnordic_connector_memory_space_v1;

typedef enum foamnordic_connector_completion_v1 {
    FOAMNORDIC_COMPLETION_SYNCHRONOUS_V1 = 0,
    FOAMNORDIC_COMPLETION_EVENT_V1 = 1,
} foamnordic_connector_completion_v1;

typedef struct foamnordic_connector_memory_api_v1 {
    uint32_t abi_version;
    uint8_t input_memory;
    uint8_t output_memory;
    uint8_t persistent_device_state;
    uint8_t completion;
    uint8_t workspace_recreated_on_shape_change;
    uint64_t preferred_batch_rows;
    uint64_t alignment_bytes;
} foamnordic_connector_memory_api_v1;

typedef struct foamnordic_connector_result_v1 {
    const void* data;
    uint64_t bytes;
    uint64_t rows;
    uint64_t columns;
    uint8_t element;
    void* owner;
} foamnordic_connector_result_v1;

typedef struct foamnordic_connector_api_v1 {
    uint32_t abi_version;
    const char* id;
    uint8_t concurrent_evaluation;
    uint8_t row_independent;
    uint8_t owns_compute_threads;
    void* (*create)(
        const void* model,
        uint64_t model_bytes,
        uint32_t threads,
        char* error,
        size_t error_capacity);
    void (*destroy)(void* model);
    int (*evaluate)(
        void* model,
        const void* input,
        uint64_t input_bytes,
        uint64_t rows,
        uint64_t columns,
        uint8_t element,
        uint64_t exchange_index,
        double physical_time,
        uint32_t rank,
        foamnordic_connector_result_v1* result,
        char* error,
        size_t error_capacity);
    void (*release)(void* owner);
} foamnordic_connector_api_v1;

typedef const foamnordic_connector_api_v1* (*foamnordic_connector_entry_v1)(void);
typedef const foamnordic_connector_memory_api_v1*
    (*foamnordic_connector_memory_entry_v1)(void);

#ifdef __cplusplus
}
#endif
