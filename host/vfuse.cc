#include "fusion.h"
#include "runtime.h"

#if PIPELINE

namespace {
uint8_t get_or_add_push_op(std::vector<detail::VectorDescRef>& inputs,
                           const detail::VectorDescRef& vec) {
  if (!vec) return PUSH_OP_BUDGET_EXCEEDED;
  if (!inputs.empty() && inputs[0] == vec) return OP_PUSH_INPUT;
  for (size_t i = 1; i < inputs.size(); ++i)
    if (inputs[i] == vec) return (uint8_t)(OP_PUSH_OPERAND_0 + (i - 1));
  if (inputs.empty()) {
    inputs.push_back(vec);
    return OP_PUSH_INPUT;
  }
  if (inputs.size() < MAX_COMBINED_INPUTS) {
    inputs.push_back(vec);
    return (uint8_t)(OP_PUSH_OPERAND_0 + (inputs.size() - 2));
  }
  return PUSH_OP_BUDGET_EXCEEDED;
}

void append_inline_scalar(std::vector<uint8_t>& rpn, uint8_t op,
                          uint32_t scalar) {
  rpn.push_back(op);
  rpn.push_back((uint8_t)(scalar & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 8) & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 16) & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 24) & 0xFF));
}

bool append_absorbed_rpn_inline(const detail::VectorDescRef& vec,
                                std::vector<detail::VectorDescRef>& inputs,
                                std::vector<uint8_t>& out) {
  if (!vec || vec->absorbed_rpn.empty() || vec->absorbed_inputs.empty())
    return false;

  for (size_t i = 0; i < vec->absorbed_rpn.size(); ++i) {
    uint8_t op = vec->absorbed_rpn[i];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = get_or_add_push_op(inputs, vec->absorbed_inputs[0]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return false;
      out.push_back(push);
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= vec->absorbed_inputs.size()) return false;
      uint8_t push = get_or_add_push_op(inputs, vec->absorbed_inputs[idx]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return false;
      out.push_back(push);
    } else if (IS_OP_SCALAR_VAR(op)) {
      if (i + 1 >= vec->absorbed_rpn.size()) return false;
      uint8_t scalar_idx = vec->absorbed_rpn[++i];
      if (scalar_idx >= vec->absorbed_scalars.size()) return false;
      append_inline_scalar(out, map_from_var_op(op),
                           vec->absorbed_scalars[scalar_idx]);
    } else if (op == OP_LOAD_INDIRECT || IS_OP_INDIRECT_UPDATE(op)) {
      return false;
    } else if (OP_INLINE_BYTES(op) > 0) {
      out.push_back(op);
      for (size_t b = 0;
           b < OP_INLINE_BYTES(op) && i + 1 < vec->absorbed_rpn.size(); ++b)
        out.push_back(vec->absorbed_rpn[++i]);
    } else {
      out.push_back(op);
    }
  }
  return true;
}
}  // namespace

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

  std::vector<detail::VectorDescRef> new_inputs;
  std::vector<uint8_t> new_rpn;
  std::vector<uint32_t> new_scalars;
  bool contains_indirect = false;
  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
        op == OP_APPLY_INDIRECT || op == OP_PUSH_INDEX) {
      contains_indirect = true;
      break;
    }
    if (OP_INLINE_BYTES(op) > 0) k += OP_INLINE_BYTES(op);
  }

  if (contains_indirect) {
    bool rewritten = false;
    for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
      uint8_t op = e->rpn_ops[k];
      if (op == OP_PUSH_INDEX && k + 2 < e->rpn_ops.size() &&
          e->rpn_ops[k + 1] == OP_LOAD_INDIRECT && e->rpn_ops[k + 2] == 0) {
        if (!append_absorbed_rpn_inline(in_vec, new_inputs, new_rpn)) return;
        rewritten = true;
        k += 2;
      } else if (op == OP_LOAD_INDIRECT) {
        return;
      } else if (op >= OP_PUSH_OPERAND_0 &&
                 op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
        size_t idx = op - OP_PUSH_OPERAND_0 + 1;
        if (idx >= e->inputs.size()) return;
        uint8_t push = get_or_add_push_op(new_inputs, e->inputs[idx]);
        if (push == PUSH_OP_BUDGET_EXCEEDED) return;
        new_rpn.push_back(push);
      } else if (op == OP_PUSH_INPUT) {
        return;
      } else if (IS_OP_SCALAR_VAR(op)) {
        new_rpn.push_back(op);
        if (k + 1 < e->rpn_ops.size()) new_rpn.push_back(e->rpn_ops[++k]);
      } else if (OP_INLINE_BYTES(op) > 0) {
        new_rpn.push_back(op);
        for (size_t b = 0; b < OP_INLINE_BYTES(op) && k + 1 < e->rpn_ops.size();
             ++b)
          new_rpn.push_back(e->rpn_ops[++k]);
      } else {
        new_rpn.push_back(op);
      }
    }
    if (!rewritten) {
      return;
    }
    new_scalars = e->scalars;
#if ENABLE_DPU_LOGGING >= 1
    DpuRuntime::get().get_logger().lock()
        << "[vfuse] inlined absorbed input into indirect consumer id=" << e->id
        << std::endl;
#else
    fprintf(stderr,
            "[vfuse] inlined absorbed input into indirect consumer id=%zu\n",
            e->id);
#endif
  } else {
    const auto& ai = in_vec->absorbed_inputs;
    size_t N = ai.size();
    size_t prefix_scalars = in_vec->absorbed_scalars.size();

    new_inputs.reserve(N + e->inputs.size() - 1);
    for (auto& v : ai) new_inputs.push_back(v);
    for (size_t k = 1; k < e->inputs.size(); ++k)
      new_inputs.push_back(e->inputs[k]);

    if (new_inputs.size() > MAX_COMBINED_INPUTS) return;

    new_scalars = in_vec->absorbed_scalars;

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
  }

  if (new_inputs.size() > MAX_COMBINED_INPUTS) return;

  // Clear absorbed state — future ops that read this vector get it from MRAM.
  auto absorbed_vec = std::move(in_vec);
  e->inputs = std::move(new_inputs);
  e->rpn_ops = std::move(new_rpn);
  e->scalars = std::move(new_scalars);
  e->is_scalar = false;
  if (absorbed_vec->last_producer_id != 0)
    e->absorbed_producer_ids.insert(absorbed_vec->last_producer_id);
  absorbed_vec->absorbed_rpn.clear();
  absorbed_vec->absorbed_scalars.clear();
  absorbed_vec->absorbed_inputs.clear();
  absorbed_vec->is_shared_intermediate = false;

  // The event that produced absorbed_vec has been inlined into e.  We want to
  // erase it from operations_ ONLY when no one else needs absorbed_vec's MRAM
  // output:
  //  - No other queued event reads it as an input.
  //  - No external holder (a live dpu_vector still bound to absorbed_vec)
  //    could submit another consumer.  The hist benchmark trips this case:
  //    `buckets` is kept on the caller's stack while N independent
  //    `sum(buckets == i)` events are submitted one at a time.  Erasing
  //    buckets' producer after the first sum leaves the next N-1 sums
  //    reading uninitialised MRAM.
  //
  // For absorbed_vec's use_count at this point we've already released e's old
  // inputs[0] ref (via the std::move above), so queue refs are: the producer's
  // output (the event we might erase) + the `absorbed_vec` local we hold here.
  // Anything more means an external dpu_vector still has it.
  bool other_consumers = false;
  for (const auto& op : operations_)
    for (const auto& inp : op->inputs)
      if (inp == absorbed_vec) {
        other_consumers = true;
        break;
      }

  size_t internal_refs = count_internal_references(absorbed_vec);
  // Allow transient queue-owned refs and temporary locals to keep the use
  // count elevated.  Only treat the vector as externally held when the count
  // stays well above the internal reference count.
  bool external_holder = absorbed_vec.use_count() > internal_refs + 5;

  if (!other_consumers) {
    for (auto it = operations_.begin(); it != operations_.end();) {
      auto& op = *it;
      bool erasable_seed = op->opcode == OP_NEGATE;
      if (op->output == absorbed_vec && op->extra_outputs.empty() &&
          (!external_holder || erasable_seed || contains_indirect)) {
        // Close the perfetto slice opened by event_enqueued so the absorbed
        // producer's track doesn't hang open for the rest of the trace.
        trace::event_fused(op, e, "");
        // The absorbed producer is never going to execute, so nothing will
        // mark it finished.  Transfer its id range into e so that when e
        // eventually completes, last_finished_id advances past the erased id
        // and anyone still waiting on it is unblocked.
        e->max_id = std::max(e->max_id, op->max_id);
        e->dependencies.insert(op->dependencies.begin(),
                               op->dependencies.end());
        // Redirect the absorbed_vec's producer so consumers that add a
        // dependency on absorbed_vec from here on wait for e, not the erased
        // producer.
        absorbed_vec->last_producer_id = e->id;
        it = operations_.erase(it);
#if ENABLE_DPU_LOGGING >= 1
        DpuRuntime::get().get_logger().lock()
            << "[vfuse] erased absorbed producer id=" << op->id
            << " for consumer id=" << e->id << std::endl;
#endif
      } else {
        ++it;
      }
    }
  }
}

// Vertical fusion: e depends on last's output (on-stack value).
// Merges e's RPN into last so both run in one kernel pass.
bool EventQueue::try_vfuse(std::shared_ptr<Event> last,
                           std::shared_ptr<Event> e) {
  if (!last->rpn_ops.empty() && IS_OP_REDUCTION(last->rpn_ops.back()))
    return false;

  // `e` may be a dpu_jit_foreach kernel using indirect ops (LOAD_INDIRECT /
  // ADD_INDIRECT / PUSH_INDEX).  Those opcodes expect the producer's output
  // to exist in MRAM for random-access loads; vfuse would absorb it
  // on-stack and leave the MRAM slot unwritten, so the indirect access
  // reads garbage (hist segfault on `sweep.py --hist`).  Also, the vfuse
  // rpn rewriter doesn't remap the single-byte operand index that follows
  // LOAD_INDIRECT, so after merging it would still reference slot 0 of a
  // combined inputs list that no longer corresponds to the absorbed vec.
  for (uint8_t op : e->rpn_ops) {
    if (op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
        op == OP_APPLY_INDIRECT || op == OP_PUSH_INDEX)
      return false;
  }

  // Safety: the on-stack value is the last chain's output.
  detail::VectorDescRef on_stack =
      last->extra_outputs.empty() ? last->output : last->extra_outputs.back();

  // If on_stack is a shared intermediate (e.g. error_shifted consumed by DIM
  // gradient chains), absorbing it on-stack would skip the MRAM write and
  // corrupt subsequent readers.  The linreg error accumulator is the one
  // exception we deliberately support here: each update forms
  //   previous_error + (dx[j] * scalar[j])
  // where the previous_error value is consumed by an ADD chain and replaced by
  // the next accumulator value.  Keeping this case fusable lets the
  // accumulator collapse into one JIT kernel while still rejecting product and
  // reduction chains that read shared materialized intermediates.
  bool e_ends_with_add = e->rpn_ops.empty() && e->opcode == OP_ADD;
  bool e_ends_with_asr_scalar =
      e->rpn_ops.empty() && e->opcode == OP_ASR_SCALAR;
  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (IS_OP_SCALAR(op)) {
      e_ends_with_asr_scalar = op == OP_ASR_SCALAR;
      k += SCALAR_INLINE_BYTES;
    } else if (IS_OP_SCALAR_VAR(op)) {
      e_ends_with_asr_scalar = op == OP_ASR_SCALAR_VAR;
      k += SCALAR_VAR_INDEX_BYTES;
    } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR ||
               op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
               op == OP_APPLY_INDIRECT) {
      k += OP_INLINE_BYTES(op);
    } else if (op == OP_ADD) {
      e_ends_with_add = true;
      e_ends_with_asr_scalar = false;
    } else if (IS_OP_BINARY(op) || IS_OP_UNARY(op) || IS_OP_REDUCTION(op) ||
               IS_OP_TERNARY(op)) {
      e_ends_with_add = false;
      e_ends_with_asr_scalar = false;
    }
  }
  bool consumes_accumulator_chain = e_ends_with_add || e_ends_with_asr_scalar;
  if (on_stack && on_stack->is_shared_intermediate &&
      !consumes_accumulator_chain)
    return false;

  // Absorbing on_stack skips its MRAM write.  Reject if any other event in
  // the queue still reads vec — they would see stale MRAM.  The deliberate
  // double-counting of e's own input refs (once via count_internal_references
  // when e is already in operations_, once via the e->inputs scan below)
  // makes this check reject retroactive vfuses that would absorb a
  // multi-consumer named output: e.g. linreg's `error_shifted` is read by
  // 10 independent reductions, so after the first reduction has been
  // folded in, the retroactive try_vfuse(accumulator, first_reduction)
  // sees lib=3 > internal=2 and bails — keeping error_shifted's MRAM
  // materialisation intact.  Non-retroactive fusion during submit passes
  // cleanly because e isn't yet in the queue.
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
  if (!consumes_accumulator_chain && !check_safety(on_stack)) return false;

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
    return PUSH_OP_BUDGET_EXCEEDED;
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
      if (push == PUSH_OP_BUDGET_EXCEEDED) {
        possible = false;
        break;
      }
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
      if (push == PUSH_OP_BUDGET_EXCEEDED) {
        possible = false;
        break;
      }
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
    } else if (OP_INLINE_BYTES(op) > 0) {
      e_mapped.push_back(op);
      for (size_t m = 0; m < OP_INLINE_BYTES(op) && k + 1 < e_rpn.size(); ++m)
        e_mapped.push_back(e_rpn[++k]);
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
  if (last_scalars.size() + e_scalars.size() > MAX_PIPELINE_SCALARS)
    return false;

  last->rpn_ops = last_rpn;
  last->rpn_ops.insert(last->rpn_ops.end(), e_mapped.begin(), e_mapped.end());
  last->scalars = last_scalars;
  last->scalars.insert(last->scalars.end(), e_scalars.begin(), e_scalars.end());
  last->inputs = combined;

  if (last->extra_outputs.empty())
    last->output = e->output;
  else
    last->extra_outputs.back() = e->output;

  // Record full merged RPN on the current chain output so future consumers can
  // inline the latest fused producer instead of waiting on the pre-fused
  // intermediate id.
  detail::VectorDescRef fused_output =
      last->extra_outputs.empty() ? last->output : last->extra_outputs.back();
  if (fused_output && !last->inputs.empty()) {
    fused_output->absorbed_rpn = last->rpn_ops;
    fused_output->absorbed_scalars = last->scalars;
    fused_output->absorbed_inputs = last->inputs;
  }

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
  last->slice_name = "Fused: [" + ops + "]";

#if ENABLE_DPU_LOGGING >= 1
  std::string rpn_dbg;
  for (size_t i = 0; i < last->rpn_ops.size(); ++i) {
    uint8_t op = last->rpn_ops[i];
    if (!rpn_dbg.empty()) rpn_dbg += " ";
    if (op == OP_PUSH_INPUT) {
      rpn_dbg += "PUSH_INPUT";
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      rpn_dbg += "PUSH_OPERAND_" + std::to_string(op - OP_PUSH_OPERAND_0);
    } else {
      std::string s = opcode_to_string(op);
      if (s.empty())
        rpn_dbg += "OP(" + std::to_string(op) + ")";
      else
        rpn_dbg += s;
      if (OP_INLINE_BYTES(op) > 0) {
        for (size_t j = 0; j < OP_INLINE_BYTES(op) && i + 1 < last->rpn_ops.size();
             ++j) {
          rpn_dbg += " " + std::to_string(last->rpn_ops[++i]);
        }
      }
    }
  }
  DpuRuntime::get().get_logger().lock()
      << "[queue-fuse] fused event id=" << e->id << " into last=" << last->id
      << " rpn=\"" << rpn_dbg << "\""
      << std::endl;
#endif
  trace::event_fused(e, last, "");
  trace::inqueue_end(e);
  return true;
}

#endif  // PIPELINE
