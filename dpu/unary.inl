#include <mram.h>

#define NEGATE(x) (-(x))
#define ABS(x) ((x) < 0 ? -(x) : (x))

#define DEFINE_UNARY_KERNEL(TYPE, OP, FUNC)                                \
  int unary_##TYPE##_##OP(void) {                                          \
    unsigned int tasklet_id = me();                                        \
    uint32_t num_elems = args.num_elements;                                \
                                                                           \
    __mram_ptr TYPE *rhs_ptr = (__mram_ptr TYPE *)(args.unary.rhs_offset); \
    __mram_ptr TYPE *res_ptr = (__mram_ptr TYPE *)(args.unary.res_offset); \
                                                                           \
    /* WRAM working buffer (DMA aligned) */                                \
    TYPE *rhs_block = (TYPE *)dpu_workspace[tasklet_id];                    \
    TYPE *res_block = (TYPE *)&dpu_workspace[tasklet_id][BLOCK_SIZE * sizeof(TYPE)]; \
                                                                           \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;               \
         block_loc < num_elems;                                            \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                  \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)         \
                                 ? (num_elems - block_loc)                 \
                                 : BLOCK_SIZE;                             \
                                                                           \
      uint32_t block_bytes = block_elems * sizeof(TYPE);                   \
                                                                           \
      /* Copy block from MRAM to WRAM */                                   \
      mram_read((__mram_ptr void const *)(rhs_ptr + block_loc), rhs_block, \
                block_bytes);                                              \
                                                                           \
      /* Compute in WRAM */                                                \
      for (uint32_t i = 0; i < block_elems; i++) {                         \
        res_block[i] = FUNC(rhs_block[i]);                                 \
      }                                                                    \
                                                                           \
      /* Write result back to MRAM */                                      \
      mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc),      \
                 block_bytes);                                             \
    }                                                                      \
    return 0;                                                              \
  }
