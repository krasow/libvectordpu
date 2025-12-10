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
    REDUCTION_OP_SUM     = 0,
    REDUCTION_OP_PRODUCT = 1,
    REDUCTION_OP_MAX     = 2,
    REDUCTION_OP_MIN     = 3
} reduction_op_t;

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

#define DEFINE_KERNELS_BY_TYPE(TYPE) \
    UNARY_KERNELS(TYPE), \
    BINARY_KERNELS(TYPE), \
    REDUCTION_KERNELS(TYPE)

#define DEFINE_ALL_KERNELS \
    DEFINE_KERNELS_BY_TYPE(INT), \
    DEFINE_KERNELS_BY_TYPE(FLOAT), \
    DEFINE_KERNELS_BY_TYPE(DOUBLE) 


typedef enum {
    DEFINE_ALL_KERNELS,

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
#undef DEFINE_KERNELS_BY_TYPE
#undef DEFINE_ALL_KERNELS

// helper to extract reduction operation from kernel ID
// assumes kernels are ordered: INT, FLOAT, DOUBLE for each op type
// very hacky but works for now
#ifdef __cplusplus
inline reduction_op_t get_reduction_op(KernelID kernel_id) {
    // There are 4 reduction ops per type (SUM, PRODUCT, MAX, MIN)
    // The reduction kernels start after all unary and binary kernels
    int reduction_base = K_REDUCTION_INT_SUM;
    int offset = static_cast<int>(kernel_id) - reduction_base;
    return static_cast<reduction_op_t>(offset % 4);
}
#endif

#endif // COMMON_H