#include <mram.h>
#define DEFINE_BINARY_KERNEL(TYPE, OP, SYMBOL)                              \
  int binary_##TYPE##_##OP(void) {                                          \
    unsigned int tasklet_id = me();                                         \
    uint32_t num_elems = args.num_elements;                                 \
                                                                            \
    __mram_ptr TYPE *lhs_ptr = (__mram_ptr TYPE *)(args.binary.lhs_offset); \
    __mram_ptr TYPE *rhs_ptr = (__mram_ptr TYPE *)(args.binary.rhs_offset); \
    __mram_ptr TYPE *res_ptr = (__mram_ptr TYPE *)(args.binary.res_offset); \
                                                                            \
    __dma_aligned TYPE lhs_block[BLOCK_SIZE];                               \
    __dma_aligned TYPE rhs_block[BLOCK_SIZE];                               \
    __dma_aligned TYPE res_block[BLOCK_SIZE];                               \
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
                                                                            \
      mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc),       \
                 block_bytes);                                              \
    }                                                                       \
    return 0;                                                               \
  }

#define INSTANTIATE_KERNELS(TYPE)         \
  DEFINE_BINARY_KERNEL(TYPE, add, +)      \
  DEFINE_BINARY_KERNEL(TYPE, subtract, -) \
  DEFINE_BINARY_KERNEL(TYPE, sum, +)      \
  DEFINE_BINARY_KERNEL(TYPE, product, *)

INSTANTIATE_KERNELS(float)
INSTANTIATE_KERNELS(int)
INSTANTIATE_KERNELS(double)

#undef INSTANTIATE_KERNELS
#undef DEFINE_BINARY_KERNEL
