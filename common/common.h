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

#define UNARY_KERNELS(TYPE) \
    K_UNARY_##TYPE##_NEGATE, \
    K_UNARY_##TYPE##_ABS

#define BINARY_KERNELS(TYPE) \
    K_BINARY_##TYPE##_ADD, \
    K_BINARY_##TYPE##_SUB

#define REDUCTION_KERNELS(TYPE) \
    K_REDUCTION_##TYPE##_SUM, \
    K_REDUCTION_##TYPE##_PRODUCT, \
    K_REDUCTION_##TYPE##_MAX, \
    K_REDUCTION_##TYPE##_MIN

#define ALL_UNARY_KERNELS \
    UNARY_KERNELS(FLOAT), \
    UNARY_KERNELS(INT), \
    UNARY_KERNELS(DOUBLE)

#define ALL_BINARY_KERNELS \
    BINARY_KERNELS(FLOAT), \
    BINARY_KERNELS(INT), \
    BINARY_KERNELS(DOUBLE)

#define ALL_REDUCTION_KERNELS \
    REDUCTION_KERNELS(FLOAT), \
    REDUCTION_KERNELS(INT), \
    REDUCTION_KERNELS(DOUBLE)

typedef enum {
    ALL_UNARY_KERNELS,
    ALL_BINARY_KERNELS,
    ALL_REDUCTION_KERNELS,

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


#undef UNARY_KERNELS
#undef BINARY_KERNELS
#undef REDUCTION_KERNELS
#undef ALL_UNARY_KERNELS
#undef ALL_BINARY_KERNELS
#undef ALL_REDUCTION_KERNELS


#endif // COMMON_H