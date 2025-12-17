#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define BLOCK_SIZE_LOG2 5              // e.g., 32 elements per block
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

typedef uint32_t KernelID;

enum KernelCategory {
    KERNEL_UNARY = 0,
    KERNEL_BINARY = 1,
    KERNEL_REDUCTION = 2
};

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