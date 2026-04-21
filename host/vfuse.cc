#include "fusion.h"
#include "runtime.h"

#if PIPELINE

// Inline the RPN of an absorbed intermediate into `e` so it can be computed
// without reading from (unwritten) MRAM.  Called from EventQueue::submit before
// the event is enqueued.
void EventQueue::expand_absorbed_inputs(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE || e->inputs.empty()) return;

  auto& in_vec = e->inputs[0];
  if (!in_vec || in_vec->absorbed_rpn.empty() ||
      in_vec->absorbed_inputs.empty())
    return;

  if (e->rpn_ops.empty()) {
    for (size_t k = 0; k < e->inputs.size(); ++k)
      e->rpn_ops.push_back(k == 0 ? OP_PUSH_INPUT
                                  : OP_PUSH_OPERAND_0 + (k - 1));
    if (e->is_scalar) {
      e->rpn_ops.push_back(map_to_var_op(e->opcode));
      e->rpn_ops.push_back(0);
      e->scalars.push_back(e->scalar_value);
    } else {
      e->rpn_ops.push_back(e->opcode);
    }
  }

  const auto& ai = in_vec->absorbed_inputs;
  size_t N = ai.size();
  size_t prefix_scalars = in_vec->absorbed_scalars.size();

  std::vector<detail::VectorDescRef> new_inputs;
  new_inputs.reserve(N + e->inputs.size() - 1);
  for (auto& v : ai) new_inputs.push_back(v);
  for (size_t k = 1; k < e->inputs.size(); ++k)
    new_inputs.push_back(e->inputs[k]);

  if (new_inputs.size() > MAX_COMBINED_INPUTS) return;

  std::vector<uint8_t> new_rpn;
  std::vector<uint32_t> new_scalars = in_vec->absorbed_scalars;

  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (op == OP_PUSH_INPUT) {
      new_rpn.insert(new_rpn.end(), in_vec->absorbed_rpn.begin(),
                     in_vec->absorbed_rpn.end());
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      uint8_t X = op - OP_PUSH_OPERAND_0;
      new_rpn.push_back(OP_PUSH_OPERAND_0 + (uint8_t)(N - 1 + X));
    } else if (IS_OP_SCALAR_VAR(op)) {
      new_rpn.push_back(op);
      if (k + 1 < e->rpn_ops.size())
        new_rpn.push_back(e->rpn_ops[++k] + (uint8_t)prefix_scalars);
    } else {
      new_rpn.push_back(op);
    }
  }
  new_scalars.insert(new_scalars.end(), e->scalars.begin(), e->scalars.end());

  // Clear absorbed state — future ops that read this vector get it from MRAM.
  auto absorbed_vec = in_vec;
  e->inputs = std::move(new_inputs);
  e->rpn_ops = std::move(new_rpn);
  e->scalars = std::move(new_scalars);
  e->is_scalar = false;
  absorbed_vec->absorbed_rpn.clear();
  absorbed_vec->absorbed_scalars.clear();
  absorbed_vec->absorbed_inputs.clear();
  absorbed_vec->is_shared_intermediate = false;

  // The event that produced absorbed_vec is now orphaned: its computation has
  // been inlined into e and no pending event reads its MRAM output.  Remove it
  // from the queue so it cannot be spuriously hfused with the growing chain
  // (which would break the absorbed_rpn tracking on the chain's output).
  bool other_consumers = false;
  for (const auto& op : operations_)
    for (const auto& inp : op->inputs)
      if (inp == absorbed_vec) {
        other_consumers = true;
        break;
      }

  if (!other_consumers) {
    operations_.erase(std::remove_if(operations_.begin(), operations_.end(),
                                     [&](const auto& op) {
                                       return op->output == absorbed_vec &&
                                              op->extra_outputs.empty();
                                     }),
                      operations_.end());
  }
}

// Vertical fusion: e depends on last's output (on-stack value).
// Merges e's RPN into last so both run in one kernel pass.
bool EventQueue::try_vfuse(std::shared_ptr<Event> last,
                           std::shared_ptr<Event> e) {
  if (!last->rpn_ops.empty() && IS_OP_REDUCTION(last->rpn_ops.back()))
    return false;

  // Safety: the on-stack value is the last chain's output.
  detail::VectorDescRef on_stack =
      last->extra_outputs.empty() ? last->output : last->extra_outputs.back();

  // If on_stack is a shared intermediate (e.g. error_shifted consumed by DIM
  // gradient chains), absorbing it on-stack would skip the MRAM write and
  // corrupt subsequent readers.
  if (on_stack && on_stack->is_shared_intermediate) return false;

  auto check_safety = [&](detail::VectorDescRef vec) {
    if (!vec) return true;
    size_t internal = 1;
    for (const auto& in : e->inputs)
      if (in == vec) internal++;
    size_t lib = count_internal_references(vec);
    for (const auto& in : e->inputs)
      if (in == vec) lib++;
    return lib <= internal;
  };
  if (!check_safety(on_stack)) return false;

  bool e_uses_on_stack = false;
  for (const auto& in : e->inputs)
    if (in == on_stack) {
      e_uses_on_stack = true;
      break;
    }
  if (!e_uses_on_stack) return false;

  for (const auto& in : e->inputs) {
    if (!in || in == on_stack) continue;
    if (in == last->output) return false;
    for (const auto& out : last->extra_outputs)
      if (in == out) return false;
  }

  std::vector<uint8_t> last_rpn;
  std::vector<uint32_t> last_scalars;
  std::vector<uint8_t> e_rpn;
  std::vector<uint32_t> e_scalars;
  build_default_rpn(last, last_rpn, last_scalars);
  build_default_rpn(e, e_rpn, e_scalars);

  std::vector<detail::VectorDescRef> combined = last->inputs;
  auto get_push_op = [&](detail::VectorDescRef vec) -> uint8_t {
    if (vec == on_stack) return PUSH_OP_ALREADY_ON_STACK;
    if (!combined.empty() && combined[0] == vec) return OP_PUSH_INPUT;
    for (size_t i = 1; i < combined.size(); ++i)
      if (combined[i] == vec) return (uint8_t)(OP_PUSH_OPERAND_0 + (i - 1));
    if (combined.size() < MAX_COMBINED_INPUTS) {
      combined.push_back(vec);
      // New element is at index (combined.size()-1); its operand slot is one
      // less (slot 0 is the primary), so operand_index = combined.size() - 2.
      return (uint8_t)(OP_PUSH_OPERAND_0 + (combined.size() - 2));
    }
    return PUSH_OP_ALREADY_ON_STACK;
  };

  auto is_commutative = [](uint8_t op) {
    return op == OP_ADD || op == OP_MUL || op == OP_EQ;
  };

  std::vector<uint8_t> e_mapped;
  bool possible = true;
  bool primary_on_stack = false;
  bool secondary_on_stack_no_primary = false;

  for (size_t k = 0; k < e_rpn.size(); ++k) {
    uint8_t op = e_rpn[k];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = get_push_op(e->inputs[0]);
      if (push != PUSH_OP_ALREADY_ON_STACK)
        e_mapped.push_back(push);
      else
        primary_on_stack = true;
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= e->inputs.size()) {
        possible = false;
        break;
      }
      uint8_t push = get_push_op(e->inputs[idx]);
      if (push == PUSH_OP_ALREADY_ON_STACK) {
        if (primary_on_stack) e_mapped.push_back(OP_DUP);
        secondary_on_stack_no_primary = true;
      } else {
        e_mapped.push_back(push);
      }
    } else if (IS_OP_SCALAR(op)) {
      e_mapped.push_back(op);
      for (int m = 0; m < SCALAR_INLINE_BYTES && k + 1 < e_rpn.size(); ++m)
        e_mapped.push_back(e_rpn[++k]);
    } else if (IS_OP_SCALAR_VAR(op)) {
      e_mapped.push_back(op);
      if (k + 1 < e_rpn.size())
        e_mapped.push_back(last_scalars.size() + e_rpn[++k]);
    } else {
      if (secondary_on_stack_no_primary && IS_OP_BINARY(op) &&
          !is_commutative(op))
        return false;
      secondary_on_stack_no_primary = false;
      e_mapped.push_back(op);
    }
  }

  if (!possible || last_rpn.size() + e_mapped.size() > MAX_VFUSE_OPS)
    return false;

  last->rpn_ops = last_rpn;
  last->rpn_ops.insert(last->rpn_ops.end(), e_mapped.begin(), e_mapped.end());
  last->scalars = last_scalars;
  last->scalars.insert(last->scalars.end(), e_scalars.begin(), e_scalars.end());
  last->inputs = combined;

  // Record full merged RPN on the absorbed output so future ops can inline it.
  if (last->extra_outputs.empty() && last->output && !last->inputs.empty()) {
    last->output->absorbed_rpn = last->rpn_ops;
    last->output->absorbed_scalars = last->scalars;
    last->output->absorbed_inputs = last->inputs;
  }

  if (last->extra_outputs.empty())
    last->output = e->output;
  else
    last->extra_outputs.back() = e->output;

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
    if (IS_OP_SCALAR(op))
      i += SCALAR_INLINE_BYTES;
    else if (IS_OP_SCALAR_VAR(op))
      i += SCALAR_VAR_INDEX_BYTES;
  }
  last->slice_name = "Fused: [" + ops + "]";

#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock()
      << "[queue-fuse] fused event id=" << e->id << " into last=" << last->id
      << std::endl;
#endif
  trace::event_fused(e, last, "");
  trace::inqueue_end(e);
  return true;
}

#endif  // PIPELINE
