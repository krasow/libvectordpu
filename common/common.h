#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define BLOCK_SIZE_LOG2 6              // 64 elements per block (256 bytes for int32)
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

#ifdef __dpu__
#include <config.h>
extern __dma_aligned uint8_t dpu_workspace[NR_TASKLETS][8 * BLOCK_SIZE];
#endif

typedef uint32_t KernelID;


enum KernelCategory {
    KERNEL_UNARY = 0,
    KERNEL_BINARY = 1,
    KERNEL_REDUCTION = 2,
    KERNEL_BINARY_SCALAR = 3
};

#include "opcodes.h"

#define MAX_PIPELINE_OPS 16
#define MAX_PIPELINE_OPERANDS 3
#define MAX_PIPELINE_STACK_DEPTH 4
#define MINIMUM_WRITE_SIZE 8

typedef struct {
    uint32_t kernel;       // 4
    uint32_t num_elements; // 4
    uint32_t size_type;    // 4
    uint8_t ktype;         // 1
    uint8_t pad[3];        // 3 (Total header size: 16 bytes)

    union {
        struct {           // binary ops
            uint32_t lhs_offset;
            uint32_t rhs_offset;
            uint32_t res_offset;
        } binary;          // 12
        struct {           // binary scalar ops
            uint32_t lhs_offset;
            uint32_t rhs_scalar;
            uint32_t res_offset;
        } binary_scalar;
        struct {           // unary ops
            uint32_t rhs_offset;
            uint32_t res_offset;
            uint32_t pad;   // pad unary to 12 bytes
        } unary;
        struct {           // reduction ops
            uint32_t rhs_offset;
            uint32_t res_offset;
        } reduction;
        struct {           // universal pipeline
            uint32_t init_offset;    // Initial input offset (LHS)
            uint32_t res_offset;     // Result offset (for vector output)
            uint32_t num_ops;
            uint8_t ops[MAX_PIPELINE_OPS];          // Fixed size buffer for opcodes
            uint32_t binary_operands[MAX_PIPELINE_OPERANDS]; // Offsets for binary operands
        } pipeline;
    };
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

#endif // COMMON_H
