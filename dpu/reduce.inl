#include <mram.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define PRODUCT(a, b) ((a) * (b))
#define SUM(a, b) ((a) + (b))

#define MINIMUM_WRITE_SIZE 8

#define DEFINE_REDUCTION_KERNEL(TYPE, OP, FUNC)                                \
  int reduction_##TYPE##_##OP(void) {                                          \
    unsigned int tasklet_id = me();                                            \
    uint32_t num_elems = args.num_elements;                                    \
                                                                               \
    __mram_ptr TYPE *rhs_ptr = (__mram_ptr TYPE *)(args.reduction.rhs_offset); \
    __mram_ptr uint64_t *res_ptr =                                             \
        (__mram_ptr uint64_t *)(args.reduction.res_offset);                    \
                                                                               \
    /* WRAM working buffer (DMA aligned) */                                    \
    __dma_aligned TYPE rhs_block[BLOCK_SIZE];                                  \
    __dma_aligned uint64_t res_block[NR_TASKLETS];                             \
                                                                               \
    TYPE local_red = (TYPE)0;                                                  \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;                   \
         block_loc < num_elems;                                                \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                      \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)             \
                                 ? (num_elems - block_loc)                     \
                                 : BLOCK_SIZE;                                 \
                                                                               \
      uint32_t block_bytes = block_elems * sizeof(TYPE);                       \
                                                                               \
      /* Copy block from MRAM to WRAM */                                       \
      mram_read((__mram_ptr void const *)(rhs_ptr + block_loc), rhs_block,     \
                block_bytes);                                                  \
                                                                               \
      /* Compute in WRAM */                                                    \
      for (uint32_t i = 0; i < block_elems; i++) {                             \
        local_red = FUNC(rhs_block[i], local_red);                             \
      }                                                                        \
    }                                                                          \
                                                                               \
    mram_write(&local_red, (__mram_ptr void *)(res_ptr + tasklet_id),          \
               MINIMUM_WRITE_SIZE);                                            \
                                                                               \
    barrier_wait(&my_barrier);                                                 \
                                                                               \
    /* Tasklet 0 performs final reduction */                                   \
    if (tasklet_id == 0) {                                                     \
      mram_read((__mram_ptr void const *)res_ptr, res_block,                   \
                NR_TASKLETS *MINIMUM_WRITE_SIZE);                              \
                                                                               \
      TYPE total = (TYPE)res_block[0];                                         \
      for (uint32_t i = 1; i < NR_TASKLETS; i++) {                             \
        total = FUNC(total, res_block[i]);                                     \
      }                                                                        \
      printf("Final reduction result in offset 0x%08x: %d\n",                  \
             (args.reduction.res_offset), (TYPE)total);                        \
                                                                               \
      mram_write(&total, (__mram_ptr void *)(args.reduction.res_offset),       \
                 MINIMUM_WRITE_SIZE);                                          \
    }                                                                          \
    return 0;                                                                  \
  }

DEFINE_REDUCTION_KERNEL(float, max, MAX)
DEFINE_REDUCTION_KERNEL(int, max, MAX)
DEFINE_REDUCTION_KERNEL(float, min, MIN)
DEFINE_REDUCTION_KERNEL(int, min, MIN)
DEFINE_REDUCTION_KERNEL(float, product, PRODUCT)
DEFINE_REDUCTION_KERNEL(int, product, PRODUCT)
DEFINE_REDUCTION_KERNEL(float, sum, SUM)
DEFINE_REDUCTION_KERNEL(int, sum, SUM)