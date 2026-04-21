#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define BLOCK_SIZE_LOG2 4              // 16 elements per block (reduced for 10-operand WRAM budget)
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

typedef uint32_t KernelID;

enum KernelCategory {
    KERNEL_UNARY = 0,
    KERNEL_BINARY = 1,
    KERNEL_REDUCTION = 2,
    KERNEL_BINARY_SCALAR = 3,
    KERNEL_PIPELINE = 4
};

#include "opcodes.h"
#include <config.h>

#ifndef MAX_VFUSE_OPS
#define MAX_VFUSE_OPS 64
#endif
#ifndef MAX_VFUSE_INPUTS
#define MAX_VFUSE_INPUTS 10
#endif
#ifndef MAX_PIPELINE_STACK_DEPTH
#define MAX_PIPELINE_STACK_DEPTH 4
#endif
#ifndef MAX_HFUSE_CHAINS
#define MAX_HFUSE_CHAINS 3
#endif

#define MINIMUM_WRITE_SIZE 8

// combined_inputs layout: [primary, operand_0, ..., operand_N].
// The primary occupies slot 0; the remaining MAX_VFUSE_INPUTS slots hold
// binary/extra operands, giving a total capacity of MAX_VFUSE_INPUTS + 1.
#define MAX_COMBINED_INPUTS (MAX_VFUSE_INPUTS + 1)

#define SCALAR_INLINE_BYTES 4
#define SCALAR_VAR_INDEX_BYTES 1

// Sentinel: value is already on the WRAM stack — do not emit a push.
#define PUSH_OP_ALREADY_ON_STACK 0xFF

// Shared WRAM workspace per tasklet:
//   input(1) + operands(MAX_VFUSE_INPUTS) + stack_buf(MAX_PIPELINE_STACK_DEPTH) + results(MAX_HFUSE_CHAINS)
#define TASKLET_WORKSPACE_SIZE \
    ((1 + MAX_VFUSE_INPUTS + MAX_PIPELINE_STACK_DEPTH + MAX_HFUSE_CHAINS) * BLOCK_SIZE * MINIMUM_WRITE_SIZE)

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
            uint8_t ops[MAX_VFUSE_OPS];          // Fixed size buffer for opcodes
            uint32_t binary_operands[MAX_VFUSE_INPUTS]; // Offsets for binary operands
            uint32_t scalars[16]; // Scalar values for scalar operators
            uint32_t extra_res_offsets[MAX_HFUSE_CHAINS];
        } pipeline;
    };
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

#if defined(__dpu__) || defined(__dpu_v1A__)
extern __dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];
extern DPU_LAUNCH_ARGS args;
#endif

#endif // COMMON_H
