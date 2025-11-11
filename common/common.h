#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define BLOCK_SIZE_LOG2 5              // e.g., 32 elements per block
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

typedef enum {
    KERNEL_UNARY     = 0,
    KERNEL_BINARY    = 1,
    KERNEL_REDUCTION = 2
} kernel_type_t;

typedef enum {
    // Unary
    K_UNARY_FLOAT_NEGATE,
    K_UNARY_FLOAT_ABS,
    K_UNARY_INT_NEGATE,
    K_UNARY_INT_ABS,

    // Binary
    K_BINARY_FLOAT_ADD,
    K_BINARY_FLOAT_SUB,
    K_BINARY_INT_ADD,
    K_BINARY_INT_SUB,

    // Reductions
    K_REDUCTION_FLOAT_SUM,
    K_REDUCTION_FLOAT_PRODUCT,
    K_REDUCTION_FLOAT_MAX,
    K_REDUCTION_FLOAT_MIN,
    K_REDUCTION_INT_SUM,
    K_REDUCTION_INT_PRODUCT,
    K_REDUCTION_INT_MAX,
    K_REDUCTION_INT_MIN,

    KERNEL_COUNT
} KernelID;

typedef struct {
    uint32_t kernel;       // 4
    uint32_t num_elements; // 4
    uint32_t size_type;    // 4

    union {
        struct {           // binary ops
            uint32_t lhs_offset;
            uint32_t rhs_offset;
            uint32_t res_offset;
        } binary;          // 12
        struct {           // unary ops
            uint32_t rhs_offset;
            uint32_t res_offset;
            uint32_t pad;   // pad unary to 12 bytes
        } unary;
        struct {           // reduction ops
            uint32_t rhs_offset;
            uint32_t res_offset;
            uint32_t pad;   // pad reduction to 12 bytes
        } reduction;
    };

    uint8_t ktype;         // 0: unary, 1: binary, 2: reduction
    uint8_t pad[7];        // pad struct to 32 bytes
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;


#endif // COMMON_H