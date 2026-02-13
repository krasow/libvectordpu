
#include <barrier.h>
#include <stdbool.h>
#include <string.h>

// STACK_DEPTH and MINIMUM_WRITE_SIZE are defined in common.h

#define DEFINE_UNIVERSAL_PIPELINE_KERNEL(TYPE)                               \
  int universal_##TYPE##_pipeline(void) {                                    \
    unsigned int id = me();                                                  \
    uint32_t n = args.num_elements, n_ops = args.pipeline.num_ops;           \
    __mram_ptr TYPE *in = (__mram_ptr TYPE *)(args.pipeline.init_offset);    \
    __mram_ptr TYPE *rs = (__mram_ptr TYPE *)(args.pipeline.res_offset);     \
    __dma_aligned TYPE st[MAX_PIPELINE_STACK_DEPTH][BLOCK_SIZE];             \
    TYPE acc;                                                                \
    bool has_r = false;                                                      \
    uint8_t r_op = 0;                                                        \
    uint32_t blk, i, b_e, b_b, sp;                                           \
                                                                             \
    for (blk = id << BLOCK_SIZE_LOG2; blk < n;                               \
         blk += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                          \
      b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;                \
      b_b = b_e * sizeof(TYPE);                                              \
      sp = 0;                                                                \
      for (uint32_t oi = 0; oi < n_ops; ++oi) {                              \
        uint8_t op = args.pipeline.ops[oi];                                  \
        if (IS_OP_STACK(op)) {                                               \
          if (sp < MAX_PIPELINE_STACK_DEPTH) {                               \
            __mram_ptr TYPE *p =                                             \
                (op == OP_PUSH_INPUT)                                        \
                    ? in                                                     \
                    : (__mram_ptr TYPE                                       \
                           *)(args.pipeline                                  \
                                  .binary_operands[op - OP_PUSH_OPERAND_0]); \
            mram_read((__mram_ptr void const *)(p + blk), st[sp++], b_b);    \
          }                                                                  \
        } else if (IS_OP_UNARY(op)) {                                        \
          TYPE *s1 = st[sp - 1];                                             \
          if (op == OP_NEGATE)                                               \
            for (i = 0; i < b_e; i++) s1[i] = NEGATE(s1[i]);                 \
          else                                                               \
            for (i = 0; i < b_e; i++) s1[i] = ABS(s1[i]);                    \
        } else if (IS_OP_BINARY(op)) {                                       \
          TYPE *s1 = st[sp - 1];                                             \
          TYPE *s2 = st[sp - 2];                                             \
          switch (op) {                                                      \
            case OP_ADD:                                                     \
              for (i = 0; i < b_e; i++) s2[i] += s1[i];                      \
              break;                                                         \
            case OP_SUB:                                                     \
              for (i = 0; i < b_e; i++) s2[i] -= s1[i];                      \
              break;                                                         \
            case OP_MUL:                                                     \
              for (i = 0; i < b_e; i++) s2[i] *= s1[i];                      \
              break;                                                         \
            case OP_DIV:                                                     \
              for (i = 0; i < b_e; i++)                                      \
                if (s1[i] != 0) s2[i] /= s1[i];                              \
              break;                                                         \
          }                                                                  \
          sp--;                                                              \
        } else {                                                             \
          /* IS_OP_REDUCTION */                                              \
          r_op = op;                                                         \
          if (!has_r) {                                                      \
            has_r = true;                                                    \
            switch (op) {                                                    \
              case OP_SUM:                                                   \
                acc = 0;                                                     \
                break;                                                       \
              case OP_PRODUCT:                                               \
                acc = 1;                                                     \
                break;                                                       \
              case OP_MIN:                                                   \
                acc = (TYPE)0x7FFFFFFF;                                      \
                break;                                                       \
              case OP_MAX:                                                   \
                acc = (TYPE)0x80000000;                                      \
                break;                                                       \
            }                                                                \
          }                                                                  \
          TYPE *s1 = st[sp - 1];                                             \
          switch (op) {                                                      \
            case OP_SUM:                                                     \
              for (i = 0; i < b_e; i++) acc += s1[i];                        \
              break;                                                         \
            case OP_PRODUCT:                                                 \
              for (i = 0; i < b_e; i++) acc *= s1[i];                        \
              break;                                                         \
            case OP_MIN:                                                     \
              for (i = 0; i < b_e; i++) {                                    \
                if (s1[i] < acc) acc = s1[i];                                \
              }                                                              \
              break;                                                         \
            case OP_MAX:                                                     \
              for (i = 0; i < b_e; i++) {                                    \
                if (s1[i] > acc) acc = s1[i];                                \
              }                                                              \
              break;                                                         \
          }                                                                  \
          sp--;                                                              \
        }                                                                    \
      }                                                                      \
      if (!has_r && sp > 0)                                                  \
        mram_write(st[sp - 1], (__mram_ptr void *)(rs + blk), b_b);          \
    }                                                                        \
    if (has_r) {                                                             \
      enum { sd = (MINIMUM_WRITE_SIZE / sizeof(TYPE)) };                     \
      uint64_t bf = 0;                                                       \
      memcpy(&bf, &acc, sizeof(TYPE));                                       \
      mram_write((void *)&bf, (__mram_ptr uint64_t *)rs + id, 8);            \
      barrier_wait(&my_barrier);                                             \
      if (id == 0) {                                                         \
        __dma_aligned TYPE rb[NR_TASKLETS * sd];                             \
        mram_read((__mram_ptr void const *)rs, rb, NR_TASKLETS * 8);         \
        TYPE tot = rb[0];                                                    \
        for (i = 1; i < NR_TASKLETS; i++) {                                  \
          TYPE v = rb[i * sd];                                               \
          if (r_op == OP_SUM)                                                \
            tot += v;                                                        \
          else if (r_op == OP_MIN) {                                         \
            if (v < tot) tot = v;                                            \
          } else if (r_op == OP_MAX) {                                       \
            if (v > tot) tot = v;                                            \
          } else                                                             \
            tot *= v;                                                        \
        }                                                                    \
        bf = 0;                                                              \
        memcpy(&bf, &tot, sizeof(TYPE));                                     \
        mram_write(&bf, (__mram_ptr void *)rs, 8);                           \
      }                                                                      \
    }                                                                        \
    return 0;                                                                \
  }
