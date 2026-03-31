#include <barrier.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// STACK_DEPTH and MINIMUM_WRITE_SIZE are defined in common.h

#define DEFINE_UNIVERSAL_PIPELINE_KERNEL(TYPE)                                \
  int universal_##TYPE##_pipeline(void) {                                     \
    unsigned int id = me();                                                   \
    uint32_t n = args.num_elements, n_ops = args.pipeline.num_ops;            \
    __mram_ptr TYPE *in_ptr = (__mram_ptr TYPE *)(args.pipeline.init_offset); \
    __mram_ptr TYPE *rs_ptr = (__mram_ptr TYPE *)(args.pipeline.res_offset);  \
                                                                              \
    /* Workspace Layout: input(0), operands(1-3), scratch(4) */               \
    TYPE *input_blk = (TYPE *)dpu_workspace[id];                              \
    TYPE *op_blks[MAX_PIPELINE_OPERANDS];                                     \
    for (int k = 0; k < MAX_PIPELINE_OPERANDS; k++)                           \
      op_blks[k] = (TYPE *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE *          \
                                              MINIMUM_WRITE_SIZE];            \
    TYPE(*scratch_blks)                                                       \
    [BLOCK_SIZE] = (TYPE(*)[BLOCK_SIZE]) &                                    \
                   dpu_workspace[id][(MAX_PIPELINE_OPERANDS + 1) *            \
                                     BLOCK_SIZE * MINIMUM_WRITE_SIZE];        \
                                                                              \
    int64_t acc_64 = 0;                                                       \
    TYPE acc;                                                                 \
    bool has_r = false;                                                       \
    uint8_t r_op = 0;                                                         \
    uint32_t blk, i, b_e, b_b, oi;                                            \
                                                                              \
    /* Pre-scan for operands and reductions */                                \
    bool uses_input = false;                                                  \
    bool uses_op[MAX_PIPELINE_OPERANDS] = {false};                            \
    oi = 0;                                                                   \
    while (oi < n_ops) {                                                      \
      uint8_t op = args.pipeline.ops[oi];                                     \
      if (IS_OP_SCALAR(op)) {                                                 \
        oi += 5; /* Opcode + 4 bytes scalar */                                \
        continue;                                                             \
      }                                                                       \
      if (op == OP_PUSH_INPUT)                                                \
        uses_input = true;                                                    \
      else if (op >= OP_PUSH_OPERAND_0 &&                                     \
               op < OP_PUSH_OPERAND_0 + MAX_PIPELINE_OPERANDS)                \
        uses_op[op - OP_PUSH_OPERAND_0] = true;                               \
      else if (IS_OP_REDUCTION(op)) {                                         \
        r_op = op;                                                            \
        has_r = true;                                                         \
      }                                                                       \
      oi++;                                                                   \
    }                                                                         \
                                                                              \
    if (has_r) {                                                              \
      switch (r_op) {                                                         \
        case OP_SUM:                                                          \
          acc_64 = 0;                                                         \
          acc = (TYPE)0;                                                      \
          break;                                                              \
        case OP_PRODUCT:                                                      \
          acc_64 = 1;                                                         \
          acc = (TYPE)1;                                                      \
          break;                                                              \
        case OP_MIN:                                                          \
          acc = (TYPE)1e30; /* Rough infinity for now */                      \
          break;                                                              \
        case OP_MAX:                                                          \
          acc = (TYPE)-1e30;                                                  \
          break;                                                              \
      }                                                                       \
    }                                                                         \
                                                                              \
    for (blk = id << BLOCK_SIZE_LOG2; blk < n;                                \
         blk += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                           \
      b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;                 \
      b_b = b_e * sizeof(TYPE);                                               \
                                                                              \
      /* 1. Fetch operands (with deduplication) */                            \
      if (uses_input)                                                         \
        mram_read((__mram_ptr void const *)(in_ptr + blk), input_blk, b_b);   \
      for (int k = 0; k < MAX_PIPELINE_OPERANDS; k++) {                       \
        if (uses_op[k]) {                                                     \
          __mram_ptr TYPE *p =                                                \
              (__mram_ptr TYPE *)(args.pipeline.binary_operands[k]);          \
          bool found = false;                                                 \
          if (uses_input && p == in_ptr) {                                    \
            op_blks[k] = input_blk;                                           \
            found = true;                                                     \
          }                                                                   \
          for (int j = 0; j < k; j++) {                                       \
            if (uses_op[j] &&                                                 \
                p == (__mram_ptr TYPE *)(args.pipeline.binary_operands[j])) { \
              op_blks[k] = op_blks[j];                                        \
              found = true;                                                   \
              break;                                                          \
            }                                                                 \
          }                                                                   \
          if (!found)                                                         \
            mram_read((__mram_ptr void const *)(p + blk), op_blks[k], b_b);   \
        }                                                                     \
      }                                                                       \
                                                                              \
      /* 2. Pointer-based Horizontal Fusion (Loop of Loops) */                \
      TYPE *st_ptr[MAX_PIPELINE_STACK_DEPTH];                                 \
      bool st_is_temp[MAX_PIPELINE_STACK_DEPTH];                              \
      uint32_t sp = 0;                                                        \
                                                                              \
      oi = 0;                                                                 \
      while (oi < n_ops) {                                                    \
        uint8_t op = args.pipeline.ops[oi];                                   \
        if (IS_OP_SCALAR(op)) {                                               \
          TYPE *s1 = st_ptr[sp - 1];                                          \
          int32_t val;                                                        \
          /* Manually copy 4 bytes to avoid alignment issues */               \
          uint8_t b0 = args.pipeline.ops[oi + 1];                             \
          uint8_t b1 = args.pipeline.ops[oi + 2];                             \
          uint8_t b2 = args.pipeline.ops[oi + 3];                             \
          uint8_t b3 = args.pipeline.ops[oi + 4];                             \
          val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));          \
          TYPE scalar = (TYPE)val;                                            \
                                                                              \
          if (!st_is_temp[sp - 1]) {                                          \
            TYPE *dest = scratch_blks[sp - 1];                                \
            switch (op) {                                                     \
              case OP_ADD_SCALAR:                                             \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] + scalar;           \
                break;                                                        \
              case OP_SUB_SCALAR:                                             \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] - scalar;           \
                break;                                                        \
              case OP_MUL_SCALAR:                                             \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] * scalar;           \
                break;                                                        \
              case OP_DIV_SCALAR:                                             \
                for (i = 0; i < b_e; i++)                                     \
                  dest[i] = (scalar != (TYPE)0) ? s1[i] / scalar : (TYPE)0;   \
                break;                                                        \
              case OP_ASR_SCALAR:                                             \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] >> scalar;          \
                break;                                                        \
            }                                                                 \
            st_ptr[sp - 1] = dest;                                            \
            st_is_temp[sp - 1] = true;                                        \
          } else {                                                            \
            switch (op) {                                                     \
              case OP_ADD_SCALAR:                                             \
                for (i = 0; i < b_e; i++) s1[i] += scalar;                    \
                break;                                                        \
              case OP_SUB_SCALAR:                                             \
                for (i = 0; i < b_e; i++) s1[i] -= scalar;                    \
                break;                                                        \
              case OP_MUL_SCALAR:                                             \
                for (i = 0; i < b_e; i++) s1[i] *= scalar;                    \
                break;                                                        \
              case OP_DIV_SCALAR:                                             \
                for (i = 0; i < b_e; i++)                                     \
                  if (scalar != (TYPE)0) s1[i] /= scalar;                     \
                break;                                                        \
              case OP_ASR_SCALAR:                                             \
                for (i = 0; i < b_e; i++) s1[i] >>= scalar;                   \
                break;                                                        \
            }                                                                 \
          }                                                                   \
          oi += 5;                                                            \
          continue;                                                           \
        }                                                                     \
        if (IS_OP_SCALAR_VAR(op)) {                                           \
          TYPE *s1 = st_ptr[sp - 1];                                          \
          uint8_t idx = args.pipeline.ops[oi + 1];                            \
          TYPE scalar = (TYPE)args.pipeline.scalars[idx];                     \
          if (!st_is_temp[sp - 1]) {                                          \
            TYPE *dest = scratch_blks[sp - 1];                                \
            switch (op) {                                                     \
              case OP_ADD_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] + scalar;           \
                break;                                                        \
              case OP_SUB_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] - scalar;           \
                break;                                                        \
              case OP_MUL_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] * scalar;           \
                break;                                                        \
              case OP_DIV_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++)                                     \
                  dest[i] = (scalar != (TYPE)0) ? s1[i] / scalar : (TYPE)0;   \
                break;                                                        \
              case OP_ASR_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] >> scalar;          \
                break;                                                        \
            }                                                                 \
            st_ptr[sp - 1] = dest;                                            \
            st_is_temp[sp - 1] = true;                                        \
          } else {                                                            \
            switch (op) {                                                     \
              case OP_ADD_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) s1[i] += scalar;                    \
                break;                                                        \
              case OP_SUB_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) s1[i] -= scalar;                    \
                break;                                                        \
              case OP_MUL_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) s1[i] *= scalar;                    \
                break;                                                        \
              case OP_DIV_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++)                                     \
                  if (scalar != (TYPE)0) s1[i] /= scalar;                     \
                break;                                                        \
              case OP_ASR_SCALAR_VAR:                                         \
                for (i = 0; i < b_e; i++) s1[i] >>= scalar;                   \
                break;                                                        \
            }                                                                 \
          }                                                                   \
          oi += 2;                                                            \
          continue;                                                           \
        }                                                                     \
        if (IS_OP_STACK(op)) {                                                \
          st_ptr[sp] = (op == OP_PUSH_INPUT)                                  \
                           ? input_blk                                        \
                           : op_blks[op - OP_PUSH_OPERAND_0];                 \
          st_is_temp[sp] = false;                                             \
          sp++;                                                               \
        } else if (IS_OP_UNARY(op)) {                                         \
          TYPE *s = st_ptr[sp - 1];                                           \
          if (!st_is_temp[sp - 1]) {                                          \
            TYPE *dest = scratch_blks[sp - 1];                                \
            if (op == OP_NEGATE)                                              \
              for (i = 0; i < b_e; i++) dest[i] = -s[i];                      \
            else                                                              \
              for (i = 0; i < b_e; i++)                                       \
                dest[i] = (s[i] < (TYPE)0) ? -s[i] : s[i];                    \
            st_ptr[sp - 1] = dest;                                            \
            st_is_temp[sp - 1] = true;                                        \
          } else {                                                            \
            if (op == OP_NEGATE)                                              \
              for (i = 0; i < b_e; i++) s[i] = -s[i];                         \
            else                                                              \
              for (i = 0; i < b_e; i++)                                       \
                s[i] = (s[i] < (TYPE)0) ? -s[i] : s[i];                       \
          }                                                                   \
        } else if (IS_OP_BINARY(op)) {                                        \
          TYPE *s1 = st_ptr[--sp];                                            \
          TYPE *s2 = st_ptr[sp - 1];                                          \
          if (!st_is_temp[sp - 1]) {                                          \
            TYPE *dest = scratch_blks[sp - 1];                                \
            switch (op) {                                                     \
              case OP_ADD:                                                    \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] + s1[i];            \
                break;                                                        \
              case OP_SUB:                                                    \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] - s1[i];            \
                break;                                                        \
              case OP_MUL:                                                    \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] * s1[i];            \
                break;                                                        \
              case OP_DIV:                                                    \
                for (i = 0; i < b_e; i++)                                     \
                  dest[i] = (s1[i] != (TYPE)0) ? s2[i] / s1[i] : (TYPE)0;     \
                break;                                                        \
              case OP_ASR:                                                    \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] >> s1[i];           \
                break;                                                        \
            }                                                                 \
            st_ptr[sp - 1] = dest;                                            \
            st_is_temp[sp - 1] = true;                                        \
          } else {                                                            \
            switch (op) {                                                     \
              case OP_ADD:                                                    \
                for (i = 0; i < b_e; i++) s2[i] += s1[i];                     \
                break;                                                        \
              case OP_SUB:                                                    \
                for (i = 0; i < b_e; i++) s2[i] -= s1[i];                     \
                break;                                                        \
              case OP_MUL:                                                    \
                for (i = 0; i < b_e; i++) s2[i] *= s1[i];                     \
                break;                                                        \
              case OP_DIV:                                                    \
                for (i = 0; i < b_e; i++)                                     \
                  if (s1[i] != (TYPE)0) s2[i] /= s1[i];                       \
                break;                                                        \
              case OP_ASR:                                                    \
                for (i = 0; i < b_e; i++) s2[i] >>= s1[i];                    \
                break;                                                        \
            }                                                                 \
          }                                                                   \
        } else { /* REDUCTION */                                              \
          TYPE *s = st_ptr[--sp];                                             \
          switch (op) {                                                       \
            case OP_SUM:                                                      \
              for (i = 0; i < b_e; i++) acc_64 += s[i];                       \
              break;                                                          \
            case OP_PRODUCT:                                                  \
              for (i = 0; i < b_e; i++) acc_64 *= s[i];                       \
              break;                                                          \
            case OP_MIN:                                                      \
              for (i = 0; i < b_e; i++)                                       \
                if (s[i] < acc) acc = s[i];                                   \
              break;                                                          \
            case OP_MAX:                                                      \
              for (i = 0; i < b_e; i++)                                       \
                if (s[i] > acc) acc = s[i];                                   \
              break;                                                          \
          }                                                                   \
        }                                                                     \
        oi++;                                                                 \
      }                                                                       \
      if (!has_r && sp > 0)                                                   \
        mram_write(st_ptr[sp - 1], (__mram_ptr void *)(rs_ptr + blk), b_b);   \
    }                                                                         \
                                                                              \
    if (has_r) {                                                              \
      bool is_sum = (r_op == OP_SUM);                                         \
      enum { sd = (MINIMUM_WRITE_SIZE / sizeof(TYPE)) };                      \
      uint64_t bf = 0;                                                        \
      if (is_sum) {                                                           \
        bf = (uint64_t)acc_64;                                                \
      } else {                                                                \
        memcpy(&bf, &acc, sizeof(TYPE));                                      \
      }                                                                       \
      extern uint64_t reduction_scratchpad[NR_TASKLETS];                      \
      reduction_scratchpad[id] = bf;                                          \
      barrier_wait(&my_barrier);                                              \
      if (id == 0) {                                                          \
        if (is_sum) {                                                         \
          int64_t tot_64 = 0;                                                 \
          uint32_t i;                                                         \
          for (i = 0; i < NR_TASKLETS; i++) {                                 \
            tot_64 += (int64_t)reduction_scratchpad[i];                       \
          }                                                                   \
          bf = (uint64_t)tot_64;                                              \
        } else {                                                              \
          TYPE res_block_tot[NR_TASKLETS * sd] __attribute__((aligned(8)));   \
          uint32_t i;                                                         \
          for (i = 0; i < NR_TASKLETS; i++) {                                 \
            res_block_tot[i * sd] = *(TYPE *)&reduction_scratchpad[i];        \
          }                                                                   \
          TYPE total = res_block_tot[0];                                      \
          for (i = 1; i < NR_TASKLETS; i++) {                                 \
            TYPE v = res_block_tot[i * sd];                                   \
            switch (r_op) {                                                   \
              case OP_SUM:                                                    \
                total += v;                                                   \
                break;                                                        \
              case OP_PRODUCT:                                                \
                total *= v;                                                   \
                break;                                                        \
              case OP_MIN:                                                    \
                if (v < total) total = v;                                     \
                break;                                                        \
              case OP_MAX:                                                    \
                if (v > total) total = v;                                     \
                break;                                                        \
            }                                                                 \
          }                                                                   \
          bf = 0;                                                             \
          memcpy(&bf, &total, sizeof(TYPE));                                  \
        }                                                                     \
        mram_write(&bf, (__mram_ptr void *)rs_ptr, MINIMUM_WRITE_SIZE);       \
      }                                                                       \
    }                                                                         \
    return 0;                                                                 \
  }
