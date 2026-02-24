#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define BLOCK_SIZE_LOG2 6              // 64 elements per block (256 bytes for int32)
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

typedef uint32_t KernelID;

enum KernelCategory {
    KERNEL_UNARY = 0,
    KERNEL_BINARY = 1,
    KERNEL_REDUCTION = 2,
    KERNEL_BINARY_SCALAR = 3
};

#include "opcodes.h"

#define MAX_PIPELINE_OPS 32
#define MAX_PIPELINE_OPERANDS 5
#define MAX_PIPELINE_STACK_DEPTH 2
#define MINIMUM_WRITE_SIZE 8

// Shared WRAM workspace for tasklets.
// Max size: input (1) + operands (MAX_PIPELINE_OPERANDS) + stack (MAX_PIPELINE_STACK_DEPTH)
#define TASKLET_WORKSPACE_SIZE ((1 + MAX_PIPELINE_OPERANDS + MAX_PIPELINE_STACK_DEPTH) * BLOCK_SIZE * MINIMUM_WRITE_SIZE)

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

#include <config.h>

#if defined(__dpu__) || defined(__dpu_v1A__)
extern __dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];
extern DPU_LAUNCH_ARGS args;
#endif

#endif // COMMON_H
