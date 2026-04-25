#include "fusion.h"
#include "runtime.h"

#if PIPELINE

// Horizontal fusion: last and e are independent chains over equal-length
// vectors.  Both run in the same kernel pass as separate WRAM chains.
bool EventQueue::try_hfuse(std::shared_ptr<Event> last,
                           std::shared_ptr<Event> e) {
  if (last->extra_outputs.size() >= MAX_HFUSE_CHAINS) return false;

  std::vector<uint8_t> last_rpn;
  std::vector<uint32_t> last_scalars;
  std::vector<uint8_t> e_rpn;
  std::vector<uint32_t> e_scalars;
  build_default_rpn(last, last_rpn, last_scalars);
  build_default_rpn(e, e_rpn, e_scalars);

  std::vector<detail::VectorDescRef> combined = last->inputs;
  auto get_push_op = [&](detail::VectorDescRef vec) -> uint8_t {
    if (!vec) return PUSH_OP_ALREADY_ON_STACK;
    if (!combined.empty() && combined[0] == vec) return OP_PUSH_INPUT;
    for (size_t i = 1; i < combined.size(); ++i)
      if (combined[i] == vec) return (uint8_t)(OP_PUSH_OPERAND_0 + (i - 1));
    if (combined.size() < MAX_COMBINED_INPUTS) {
      combined.push_back(vec);
      return (uint8_t)(OP_PUSH_OPERAND_0 + (combined.size() - 2));
    }
    return PUSH_OP_BUDGET_EXCEEDED;
  };

  std::vector<uint8_t> e_mapped;
  e_mapped.push_back(OP_NEXT_CHAIN);
  bool possible = true;

  for (size_t k = 0; k < e_rpn.size(); ++k) {
    uint8_t op = e_rpn[k];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = get_push_op(e->inputs[0]);
      if (push == PUSH_OP_ALREADY_ON_STACK ||
          push == PUSH_OP_BUDGET_EXCEEDED) {
        possible = false;
        break;
      }
      e_mapped.push_back(push);
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= e->inputs.size()) {
        possible = false;
        break;
      }
      uint8_t push = get_push_op(e->inputs[idx]);
      if (push == PUSH_OP_ALREADY_ON_STACK ||
          push == PUSH_OP_BUDGET_EXCEEDED) {
        possible = false;
        break;
      }
      e_mapped.push_back(push);
    } else if (IS_OP_SCALAR(op)) {
      e_mapped.push_back(op);
      for (int m = 0; m < SCALAR_INLINE_BYTES && k + 1 < e_rpn.size(); ++m)
        e_mapped.push_back(e_rpn[++k]);
    } else if (IS_OP_SCALAR_VAR(op)) {
      e_mapped.push_back(op);
      if (k + 1 < e_rpn.size())
        e_mapped.push_back(last_scalars.size() + e_rpn[++k]);
    } else if (OP_INLINE_BYTES(op) > 0) {
      e_mapped.push_back(op);
      for (size_t m = 0; m < OP_INLINE_BYTES(op) && k + 1 < e_rpn.size(); ++m)
        e_mapped.push_back(e_rpn[++k]);
    } else {
      e_mapped.push_back(op);
    }
  }

  if (!possible || last_rpn.size() + e_mapped.size() > MAX_VFUSE_OPS)
    return false;
  if (last_scalars.size() + e_scalars.size() > MAX_PIPELINE_SCALARS)
    return false;

  last->rpn_ops = last_rpn;
  last->rpn_ops.insert(last->rpn_ops.end(), e_mapped.begin(), e_mapped.end());
  last->scalars = last_scalars;
  last->scalars.insert(last->scalars.end(), e_scalars.begin(), e_scalars.end());
  last->inputs = combined;
  last->extra_outputs.push_back(e->output);

  last->max_id = std::max(last->max_id, e->id);
  last->kid = last->pipeline_kid;
  for (const auto& in : e->inputs)
    if (in && in->last_producer_id != 0 && in->last_producer_id != last->id)
      last->dependencies.insert(in->last_producer_id);
  if (e->output) e->output->last_producer_id = last->id;
  for (auto& out : e->extra_outputs)
    if (out) out->last_producer_id = last->id;

  std::string ops;
  for (size_t i = 0; i < last->rpn_ops.size(); ++i) {
    uint8_t op = last->rpn_ops[i];
    std::string s = opcode_to_string(op);
    if (s.empty()) continue;
    if (!ops.empty()) ops += ", ";
    ops += s;
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }
  last->slice_name = "Horiz-Fused: [" + ops + "]";

#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock()
      << "[queue-fuse] horizontally fused event id=" << e->id
      << " into last=" << last->id << std::endl;
#endif
  trace::event_fused(e, last, "");
  trace::inqueue_end(e);
  return true;
}

#endif  // PIPELINE
