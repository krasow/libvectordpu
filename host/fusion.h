#pragma once
// Internal helpers shared by vfuse.cc, hfuse.cc, and jit.cc.
// Not part of the public API.

#include <cstdint>

#include "common.h"
#include "opinfo.h"
#include "perfetto/trace.h"
#include "perfetto/trace_internal.h"
#include "queue.h"

#if PIPELINE
inline uint8_t map_to_var_op(uint8_t op) {
  switch (op) {
    case OP_ADD_SCALAR:
      return OP_ADD_SCALAR_VAR;
    case OP_SUB_SCALAR:
      return OP_SUB_SCALAR_VAR;
    case OP_MUL_SCALAR:
      return OP_MUL_SCALAR_VAR;
    case OP_DIV_SCALAR:
      return OP_DIV_SCALAR_VAR;
    case OP_ASR_SCALAR:
      return OP_ASR_SCALAR_VAR;
    case OP_EQ_SCALAR:
      return OP_EQ_SCALAR_VAR;
    case OP_LT_SCALAR:
      return OP_LT_SCALAR_VAR;
    default:
      return op;
  }
}

// Expand a raw (unfused) event's opcode into its canonical RPN sequence.
inline void build_default_rpn(const std::shared_ptr<Event>& e,
                              std::vector<uint8_t>& rpn,
                              std::vector<uint32_t>& scalars) {
  rpn = e->rpn_ops;
  scalars = e->scalars;
  if (!rpn.empty()) return;
  if (!e->inputs.empty()) rpn.push_back(OP_PUSH_INPUT);
  for (size_t i = 1; i < e->inputs.size(); ++i)
    rpn.push_back(OP_PUSH_OPERAND_0 + (i - 1));
  if (e->is_scalar) {
    rpn.push_back(map_to_var_op(e->opcode));
    rpn.push_back(0);
    scalars.push_back(e->scalar_value);
  } else {
    rpn.push_back(e->opcode);
  }
}
#endif
