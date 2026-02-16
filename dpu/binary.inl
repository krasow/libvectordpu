#include <mram.h>
#define DEFINE_BINARY_KERNEL(TYPE, OP, SYMBOL)                              \
  int binary_##TYPE##_##OP(void) {                                          \
    unsigned int tasklet_id = me();                                         \
    uint32_t num_elems = args.num_elements;                                 \
    __mram_ptr TYPE *lhs_ptr = (__mram_ptr TYPE *)(args.binary.lhs_offset); \
    __mram_ptr TYPE *rhs_ptr = (__mram_ptr TYPE *)(args.binary.rhs_offset); \
    __mram_ptr TYPE *res_ptr = (__mram_ptr TYPE *)(args.binary.res_offset); \
                                                                            \
    TYPE *lhs_block = (TYPE *)dpu_workspace[tasklet_id];                    \
    TYPE *rhs_block =                                                       \
        (TYPE *)&dpu_workspace[tasklet_id][BLOCK_SIZE * sizeof(TYPE)];      \
    TYPE *res_block =                                                       \
        (TYPE *)&dpu_workspace[tasklet_id][2 * BLOCK_SIZE * sizeof(TYPE)];  \
                                                                            \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;                \
         block_loc < num_elems;                                             \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                   \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)          \
                                 ? (num_elems - block_loc)                  \
                                 : BLOCK_SIZE;                              \
      uint32_t block_bytes = block_elems * sizeof(TYPE);                    \
                                                                            \
      mram_read((__mram_ptr void const *)(lhs_ptr + block_loc), lhs_block,  \
                block_bytes);                                               \
      mram_read((__mram_ptr void const *)(rhs_ptr + block_loc), rhs_block,  \
                block_bytes);                                               \
                                                                            \
      for (uint32_t i = 0; i < block_elems; i++) {                          \
        res_block[i] = lhs_block[i] SYMBOL rhs_block[i];                    \
      }                                                                     \
      mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc),       \
                 block_bytes);                                              \
    }                                                                       \
    return 0;                                                               \
  }
#define DEFINE_BINARY_SCALAR_KERNEL(TYPE, OP, SYMBOL)                      \
  int binary_scalar_##TYPE##_##OP(void) {                                  \
    unsigned int tasklet_id = me();                                        \
    uint32_t num_elems = args.num_elements;                                \
    __mram_ptr TYPE *lhs_ptr =                                             \
        (__mram_ptr TYPE *)(args.binary_scalar.lhs_offset);                \
    TYPE rhs_val = (TYPE)(args.binary_scalar.rhs_scalar);                  \
    __mram_ptr TYPE *res_ptr =                                             \
        (__mram_ptr TYPE *)(args.binary_scalar.res_offset);                \
                                                                           \
    TYPE *lhs_block = (TYPE *)dpu_workspace[tasklet_id];                   \
    TYPE *res_block =                                                      \
        (TYPE *)&dpu_workspace[tasklet_id][BLOCK_SIZE * sizeof(TYPE)];     \
                                                                           \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;               \
         block_loc < num_elems;                                            \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                  \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)         \
                                 ? (num_elems - block_loc)                 \
                                 : BLOCK_SIZE;                             \
      uint32_t block_bytes = block_elems * sizeof(TYPE);                   \
                                                                           \
      mram_read((__mram_ptr void const *)(lhs_ptr + block_loc), lhs_block, \
                block_bytes);                                              \
                                                                           \
      for (uint32_t i = 0; i < block_elems; i++) {                         \
        res_block[i] = lhs_block[i] SYMBOL rhs_val;                        \
      }                                                                    \
                                                                           \
      mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc),      \
                 block_bytes);                                             \
    }                                                                      \
    return 0;                                                              \
  }
