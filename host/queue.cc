#include "queue.h"

#include <cassert>
#include <mutex>
#include <ostream>
#include <thread>

#include "jit.h"
#include "opinfo.h"
#include "perfetto/trace_internal.h"
#include "runtime.h"
#include "vectordpu.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

/*static*/ dpu_error_t upmem_callback([[maybe_unused]] struct dpu_set_t stream,
                                      [[maybe_unused]] uint32_t rank_id,
                                      void* data) {
  auto self_ptr = static_cast<std::shared_ptr<Event>*>(data);
  std::shared_ptr<Event> me = *self_ptr;

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  auto& events = queue.get_active_events();
  std::mutex& mtx = queue.get_mutex();

  {
    std::lock_guard<std::mutex> lock(mtx);
    me->mark_finished();

    while (!events.empty() && events.front()->finished) {
      // The event at the front of the list is the one currently "active"
      auto e = events.front();
      // Mark this event (and any events merged into it) as finished.
      // This tells the host that it's safe to read the results of all merged
      // ops.
      queue.last_finished_id_.store(e->max_id);
      trace::execution_end();
      events.pop_front();

      // If there's a next event, start its trace slice.
      if (!events.empty()) {
        auto next = events.front();
        trace::execution_begin(next);
        // If 'next' is also finished, the loop will continue and end its trace
        // slice.
      }
    }
  }

  static std::atomic<size_t> callback_count{0};
  size_t count = ++callback_count;
  if (count % 100 == 0) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[queue-heartbeat] callback fired (" << count
                  << ") for id=" << me->id << std::endl;
#endif
  }

  delete self_ptr;
  queue.outstanding_callbacks_--;

  return DPU_OK;
}

void Event::add_completion_callback(std::shared_ptr<Event> self) {
  assert(this->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto wrapper = new std::shared_ptr<Event>(self);

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

void EventQueue::add_fence(std::shared_ptr<Event> e) {
  assert(e->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto wrapper = new std::shared_ptr<Event>(std::move(e));

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

constexpr bool NO_PROGRESS = false;
constexpr bool YES_PROGRESS = true;

bool EventQueue::process_next() {
  std::shared_ptr<Event> e;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (operations_.empty()) {
      return NO_PROGRESS;
    }
    e = operations_.front();
    operations_.pop_front();
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
#endif

#if ENABLE_DPU_LOGGING >= 1
  logger.lock() << "[event-logger] id=" << e->id
                << " type=" << operationtype_to_string(e->op)
                << " phase=started" << std::endl;
#endif

  trace::inqueue_end(e);

  // Wait for dependencies
  if (!e->dependencies.empty()) {
    size_t max_dep = 0;
    for (size_t dep : e->dependencies) {
      if (dep > max_dep) max_dep = dep;
    }
#if ENABLE_DPU_LOGGING >= 1
    if (this->get_last_finished_id() < max_dep) {
      logger.lock() << "[queue-wait] id=" << e->id
                    << " waiting for max_dep=" << max_dep
                    << " (current=" << this->get_last_finished_id() << ")"
                    << std::endl;
    }
#endif
    size_t loop_count = 0;
    while (this->get_last_finished_id() < max_dep) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
#if ENABLE_DPU_LOGGING >= 1
      if (++loop_count % 1000 == 0) {
        logger.lock() << "[queue-heartbeat] id=" << e->id
                      << " dependency block on " << max_dep
                      << " (current=" << this->get_last_finished_id() << ")"
                      << std::endl;
      }
#endif
    }
  }

  if (e->slice_name.empty()) {
    e->slice_name = operationtype_to_string(e->op);
    if (e->op == Event::OperationType::COMPUTE) {
      e->slice_name = kernel_id_to_string(e->kid);
      if (!e->rpn_ops.empty()) {
        e->slice_name += " (Fused)";
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (running_events_.empty()) {
      trace::execution_begin(e);
    }
    running_events_.push_back(e);
    current_event_ = e;

    trace::active_ops_counter(running_events_.size());
  }

#if JIT
  if (e->op == Event::OperationType::COMPUTE && MAX_JIT_QUEUE_DEPTH > 0) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (!e->is_locked_for_jit) {
        lock_for_jit(e);
      }

      // Opportunistic batching: if we are about to flush, try to pull more
      // compute events from the queue first.
      if (!e->jit_future.valid()) {
        // If the queue is empty, give the submitter a tiny bit of time to add
        // more compute events so we can mega-batch them. 200us is negligible
        // compared to JIT compilation time (~100ms+) but enough to catch a loop.
        if (operations_.empty()) {
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }

        auto it = operations_.begin();
        while (pending_unique_kernels_.size() < MAX_JIT_QUEUE_DEPTH &&
               it != operations_.end()) {
          auto next = *it;
          if (next->op != Event::OperationType::COMPUTE) break;

          lock_for_jit(next);
          if (e->jit_future.valid()) break;
          ++it;
        }

        // Final flush if still not valid
        if (!e->jit_future.valid()) {
          flush_jit_batch();
        }
      }
    }
    // Await background compilation
    if (e->jit_future.valid()) {
#if ENABLE_DPU_LOGGING >= 1
      Logger& logger = DpuRuntime::get().get_logger();
      logger.lock() << "[queue-jit] Awaiting background JIT compilation for id="
                    << e->id << std::endl;
#endif
      e->jit_binary_path = e->jit_future.get();
    }
  }
#endif

  // JIT Binary Handling
  std::string required_binary;
  if (!e->jit_binary_path.empty()) {
    required_binary = e->jit_binary_path;
  } else if (e->op == Event::OperationType::COMPUTE) {
    // Compute events without a JIT path must use the default binary
    required_binary = DpuRuntime::get().get_default_binary_path();
  } else {
    // Non-compute events (transfers, fences) can stay on the current binary
    if (current_binary_path_.empty()) {
      required_binary = DpuRuntime::get().get_default_binary_path();
    } else {
      required_binary = current_binary_path_;
    }
  }

  // Check if we need to switch binaries
  if (!required_binary.empty() && required_binary != current_binary_path_) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[queue-jit] Switching binary to " << required_binary
                  << " (was " << current_binary_path_ << ")" << std::endl;
#endif
    trace::jit_binary_switch(current_binary_path_, required_binary);

    // Wait for all running events to complete before swapping
    // We need to wait until running_events_ is empty (except for current 'e'
    // which is in it now) Actually 'e' is in running_events_ now. We should
    // probably wait for *other* events. But 'e' hasn't started on DPU yet.

    // Spin wait for previous events to finish
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        if (running_events_.size() <= 1) {  // Only 'e' is in running_events_
          // Sanity check: e must be the only one
          if (running_events_.front() == e) break;
        }
      }
      std::this_thread::yield();
    }

    // Load new binary
    dpu_set_t& dpu_set = DpuRuntime::get().dpu_set();
    DPU_ASSERT(dpu_load(dpu_set, required_binary.c_str(), nullptr));
    current_binary_path_ = required_binary;
  } else if (current_binary_path_.empty()) {
    // First run, assumed initialized by runtime but let's track it
    current_binary_path_ = DpuRuntime::get().get_default_binary_path();
  }

  try {
    switch (e->op) {
      case Event::OperationType::FENCE:
        this->add_fence(e);
        break;
      case Event::OperationType::COMPUTE:
        e->started = true;
#if PIPELINE
        if (!e->rpn_ops.empty() || e->is_locked_for_jit) {
          // Determine Kernel ID: use jit_sub_kernel_idx for JIT batched
          // kernels, otherwise regular e->kid
          KernelID dynamic_kid =
              e->is_locked_for_jit ? e->jit_sub_kernel_idx : e->kid;
          // Automatic fusion or manual RPN pipeline
          detail::internal_launch_universal_pipeline(
              e->output, (e->inputs.empty() ? nullptr : e->inputs[0]),
              e->rpn_ops,
              (e->inputs.size() > 1
                   ? std::vector<detail::VectorDescRef>(e->inputs.begin() + 1,
                                                        e->inputs.end())
                   : std::vector<detail::VectorDescRef>()),
              dynamic_kid);
        } else if (e->cb) {
          e->cb();
        }
#else
        if (e->cb) {
          e->cb();
        }
#endif
        e->add_completion_callback(e);
        break;
      case Event::OperationType::DPU_TRANSFER:
        e->started = true;
        e->cb();
        e->add_completion_callback(e);
        break;
      case Event::OperationType::HOST_TRANSFER:
        e->started = true;
        e->cb();
        e->add_completion_callback(e);
        break;
      default:
        assert(false && "Unknown event type");
    }
  } catch (const DpuOOMException& ex) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[queue-oom] OOM detected for event id=" << e->id
                  << ". Throttling..." << std::endl;
#endif
    // reset event state
    e->started = false;

    // remove from running_events_ and push back to front of operations_
    {
      std::lock_guard<std::mutex> lock(mtx_);
      running_events_.remove(e);
      operations_.push_front(e);
      // reset current event since we are backing off
      if (current_event_ == e) current_event_ = nullptr;
    }
    // no progress as we caught an exception
    return NO_PROGRESS;
  }

  debug_active_events();
  debug_print_queue();
  return YES_PROGRESS;
}

void EventQueue::process_events(size_t wait_for_id) {
  while (true) {
    bool progress = this->process_next();

    if (this->get_last_finished_id() >= wait_for_id) {
      break;
    }

    // exit if no more events to process and everything is finished
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (operations_.empty() && running_events_.empty()) break;
    }
    if (!progress) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    static size_t loop_count = 0;
    if (++loop_count % 1000 == 0) {
#if ENABLE_DPU_LOGGING >= 1
      Logger& logger = DpuRuntime::get().get_logger();
      std::lock_guard<std::mutex> lock(mtx_);
      logger.lock() << "[queue-heartbeat] process_events waiting for "
                    << wait_for_id
                    << " (last_finished=" << this->get_last_finished_id()
                    << " ops=" << operations_.size()
                    << " running=" << running_events_.size() << ")"
                    << std::endl;
#endif
    }
  }
}

void EventQueue::debug_print_queue() {
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  if (!operations_.empty()) {
    logger.lock() << "[EventQueue] Current queue state:" << std::endl;

    std::deque<std::shared_ptr<Event>> temp_queue = operations_;

    int i = 0;
    while (!temp_queue.empty()) {
      auto e = temp_queue.front();  // Get the front element
      logger.lock() << "\t\t" << i << ". id=" << e->id
                    << " type=" << operationtype_to_string(e->op)
                    << " started=" << e->started << " finished=" << e->finished
                    << std::endl;
      temp_queue.pop_front();  // Pop the element from the temporary queue
      i++;
    }
  } else {
    logger.lock() << "[EventQueue] Queue is empty." << std::endl;
  }
#endif
}

void EventQueue::debug_active_events() {
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();

  auto& events = get_active_events();
  std::mutex& events_mutex = get_mutex();
  {
    std::lock_guard<std::mutex> lock(events_mutex);
    if (!events.empty()) {
      logger.lock() << "[EventQueue] Current active events:" << std::endl;

      int i = 0;
      for (const auto& e : events) {
        logger.lock() << "\t\t" << i << ". id=" << e->id
                      << " type=" << operationtype_to_string(e->op)
                      << " started=" << e->started
                      << " finished=" << e->finished << std::endl;
        i++;
      }
    } else {
      logger.lock() << "[EventQueue] No active events." << std::endl;
    }
  }
#endif
}

#if JIT
void EventQueue::flush_jit_batch() {
  if (pending_unique_kernels_.empty()) return;

  std::vector<std::pair<std::vector<uint8_t>, std::string>> batch =
      pending_unique_kernels_;

  // Create an async task
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-jit] Flushing " << batch.size()
                << " kernels to async JIT compiler." << std::endl;
#endif

  std::shared_future<std::string> future =
      std::async(std::launch::async, [batch]() { return jit_compile(batch); });

  // Assign future to all pending events
  for (auto ev : pending_jit_events_) {
    ev->jit_future = future;
  }

  // Clear pending events but KEEP pending_unique_kernels_ 
  pending_jit_events_.clear();
}

void EventQueue::lock_for_jit(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE) return;
  if (e->is_locked_for_jit) return;

  e->is_locked_for_jit = true;

  // 1. Ensure it has an RPN sequence (even single operations need it for the
  // batched Kernel)
  if (e->rpn_ops.empty()) {
    if (e->is_scalar) {
      e->rpn_ops.push_back(OP_PUSH_INPUT);
      e->rpn_ops.push_back(e->opcode);
      const uint8_t* p = reinterpret_cast<const uint8_t*>(&e->scalar_value);
      e->rpn_ops.insert(e->rpn_ops.end(), p, p + sizeof(uint32_t));
    } else {
      e->rpn_ops.push_back(OP_PUSH_INPUT);
      if (e->inputs.size() > 1) {
        e->rpn_ops.push_back(OP_PUSH_OPERAND_0);
      }
      e->rpn_ops.push_back(e->opcode);
    }
  }

  // 2. Identify type name
  const char* type_name = "int32_t";  // Default
  if (e->output && e->output->type_name) {
    std::string tn = e->output->type_name;
    if (tn == "i" || tn == "int")
      type_name = "int32_t";
    else if (tn == "j" || tn == "uint32_t")
      type_name = "uint32_t";
    else if (tn == "f" || tn == "float")
      type_name = "float";
    else if (tn == "d" || tn == "double")
      type_name = "double";
    else
      type_name = e->output->type_name;
  }

  std::pair<std::vector<uint8_t>, std::string> signature = {e->rpn_ops,
                                                            type_name};

  // 3. Find if this kernel exists in the current pending batch
  int idx = -1;
  for (size_t i = 0; i < pending_unique_kernels_.size(); ++i) {
    if (pending_unique_kernels_[i] == signature) {
      idx = i;
      break;
    }
  }

  if (idx != -1) {
    // Found a match in the current batch!
    e->jit_sub_kernel_idx = idx;
    pending_jit_events_.push_back(e);
    return;
  }

  // 4. Cumulative Expansion Threshold (Sticky Mega-Batching)
  // If we need to add a NEW unique kernel but the set is full, start a new
  // epoch.
  if (pending_unique_kernels_.size() >= MAX_JIT_QUEUE_DEPTH) {
    flush_jit_batch();
    pending_unique_kernels_.clear();
  }

  e->jit_sub_kernel_idx = pending_unique_kernels_.size();
  pending_unique_kernels_.push_back(signature);
  pending_jit_events_.push_back(e);

  // 5. Automatic flush when adding a new signature to ensure its future is
  // available to following logic.
  flush_jit_batch();
}
#endif

void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::mutex> lock(mtx_);
  // Backpressure: block if total in-flight events (pending + running) exceed
  // limit. Running events hold shared_ptr<VectorDesc> references that prevent
  // DPU MRAM deallocation, so we must count them too—not just the pending
  // operations_ queue.
  while (operations_.size() + running_events_.size() >= max_queue_depth_) {
    mtx_.unlock();
    // Actively drain: process pending events so their callbacks can fire,
    // releasing VectorDescRef shared_ptrs and freeing DPU MRAM.
    this->process_next();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mtx_.lock();
  }

  e->id = counter_++;
  e->max_id = e->id;

  trace::event_enqueued(e, operations_, running_events_);
  trace::active_ops_counter(operations_.size());

  bool fused = false;
#if PIPELINE
  if (e->op == Event::OperationType::COMPUTE && !operations_.empty()) {
    auto last = operations_.back();
    if (last->op == Event::OperationType::COMPUTE && last->output != nullptr) {
      // Look for dependency: does any input of 'e' match 'last->output'?
      bool dependent = false;
      for (size_t i = 0; i < e->inputs.size(); ++i) {
        if (e->inputs[i] == last->output) {
          dependent = true;
          break;
        }
      }

      if (dependent) {
        // Check fusion constraints:
        // 1. Total ops <= MAX_PIPELINE_OPS
        // 2. Can we map inputs? (max MAX_PIPELINE_OPERANDS binary operands)
        if (dependent) {
          // Promote 'last' RPN if needed to estimate new size
          // Promote 'last' RPN if needed to estimate new size
          std::vector<uint8_t> last_rpn = last->rpn_ops;
          if (last_rpn.empty()) {
            if (last->is_scalar) {
              last_rpn.push_back(OP_PUSH_INPUT);
              last_rpn.push_back(last->opcode);
              const uint8_t* p =
                  reinterpret_cast<const uint8_t*>(&last->scalar_value);
              last_rpn.insert(last_rpn.end(), p, p + sizeof(uint32_t));
            } else {
              last_rpn.push_back(OP_PUSH_INPUT);
              if (last->inputs.size() > 1)
                last_rpn.push_back(OP_PUSH_OPERAND_0);
              last_rpn.push_back(last->opcode);
            }
          }

          // Promote 'e' RPN if needed
          std::vector<uint8_t> e_rpn = e->rpn_ops;
          if (e_rpn.empty()) {
            if (e->is_scalar) {
              if (!e->inputs.empty()) e_rpn.push_back(OP_PUSH_INPUT);
              e_rpn.push_back(e->opcode);
              const uint8_t* p =
                  reinterpret_cast<const uint8_t*>(&e->scalar_value);
              e_rpn.insert(e_rpn.end(), p, p + sizeof(uint32_t));
            } else {
              if (!e->inputs.empty()) e_rpn.push_back(OP_PUSH_INPUT);
              if (e->inputs.size() > 1) e_rpn.push_back(OP_PUSH_OPERAND_0);
              e_rpn.push_back(e->opcode);
            }
          }

          std::vector<detail::VectorDescRef> combined_inputs = last->inputs;
          auto get_operand_push_op = [&](detail::VectorDescRef vec) -> uint8_t {
            if (vec == last->output) return 0;  // Already on stack
            if (vec == combined_inputs[0]) return OP_PUSH_INPUT;
            for (size_t i = 1; i < combined_inputs.size(); ++i) {
              if (combined_inputs[i] == vec) return OP_PUSH_OPERAND_0 + (i - 1);
            }
            if (combined_inputs.size() < MAX_PIPELINE_OPERANDS + 1) {
              combined_inputs.push_back(vec);
              return OP_PUSH_OPERAND_0 + (combined_inputs.size() - 2);
            }
            return 0xFF;  // Too many unique operands
          };

          std::vector<uint8_t> e_rpn_mapped;
          bool possible = true;
          for (size_t k = 0; k < e_rpn.size(); ++k) {
            uint8_t op = e_rpn[k];
            if (IS_OP_SCALAR(op)) {
              e_rpn_mapped.push_back(op);
              for (int m = 0; m < 4; ++m) {
                if (++k < e_rpn.size()) e_rpn_mapped.push_back(e_rpn[k]);
              }
            } else if (op == OP_PUSH_INPUT) {
              uint8_t push_op = get_operand_push_op(e->inputs[0]);
              if (push_op == 0xFF) {
                possible = false;
                break;
              }
              if (push_op != 0) e_rpn_mapped.push_back(push_op);
            } else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
              uint32_t orig_idx = op - OP_PUSH_OPERAND_0;
              uint8_t push_op = get_operand_push_op(e->inputs[orig_idx + 1]);
              if (push_op == 0xFF) {
                possible = false;
                break;
              }
              if (push_op != 0) e_rpn_mapped.push_back(push_op);
            } else {
              e_rpn_mapped.push_back(op);
            }
          }

          if (possible &&
              (last_rpn.size() + e_rpn_mapped.size() > MAX_PIPELINE_OPS)) {
            possible = false;
          }

          if (possible) {
            if (last->rpn_ops.empty()) {
              last->rpn_ops = last_rpn;
              last->kid = last->pipeline_kid;
            }
            last->rpn_ops.insert(last->rpn_ops.end(), e_rpn_mapped.begin(),
                                 e_rpn_mapped.end());
            last->inputs = combined_inputs;
            last->output = e->output;
            last->max_id = std::max(last->max_id, e->id);
            fused = true;

            // Update dependencies for fused event
            for (const auto& in : e->inputs) {
              if (in && in->last_producer_id != 0 &&
                  in->last_producer_id != last->id) {
                last->dependencies.insert(in->last_producer_id);
              }
            }
            // Update last producer for the NEW output
            if (last->output) {
              last->output->last_producer_id = last->id;
            }

            // Build descriptive slice name for fused event
            std::string ops_list;
            for (size_t i = 0; i < last->rpn_ops.size(); ++i) {
              uint8_t op = last->rpn_ops[i];
              std::string s = opcode_to_string(op);
              if (s.empty()) continue;
              if (!ops_list.empty()) ops_list += ", ";
              ops_list += s;
              if (IS_OP_SCALAR(op)) {
                i += sizeof(uint32_t);
              }
            }
            last->slice_name = "Fused: [" + ops_list + "]";

#if ENABLE_DPU_LOGGING >= 1
            Logger& logger = DpuRuntime::get().get_logger();
            logger.lock() << "[queue-fuse] fused event id=" << e->id
                          << " into last=" << last->id
                          << " new_ops_count=" << last->rpn_ops.size()
                          << std::endl;
#endif
            std::string fused_ops;
            for (size_t i = 0; i < e_rpn_mapped.size(); ++i) {
              uint8_t op = e_rpn_mapped[i];
              std::string s = opcode_to_string(op);
              if (s.empty()) continue;
              if (!fused_ops.empty()) fused_ops += ", ";
              fused_ops += s;
              if (IS_OP_SCALAR(op)) {
                i += sizeof(uint32_t);
              }
            }

            trace::event_fused(e, last, fused_ops);
            trace::inqueue_end(e);
          }
        }
      }
    }
  }
#endif

  if (!fused) {
#if JIT
    // Lock the previous event for JIT since it won't be fused into anymore
    if (!operations_.empty()) {
      auto last = operations_.back();
      if (last->op == Event::OperationType::COMPUTE &&
          !last->is_locked_for_jit) {
        lock_for_jit(last);
      }
    }
    // Automatically promote all single compute events to batched kernels to
    // avoid binary thrashing
    if (e->op == Event::OperationType::COMPUTE && e->rpn_ops.empty() &&
        MAX_JIT_QUEUE_DEPTH > 0) {
      e->kid = e->pipeline_kid;  // Ensure it behaves like a pipeline payload
    }
#endif
    // Identify producers for all inputs
    for (const auto& in : e->inputs) {
      if (in && in->last_producer_id != 0) {
        e->dependencies.insert(in->last_producer_id);
      }
    }
    // Set this event as the last producer for the output
    if (e->output) {
      e->output->last_producer_id = e->id;
    }

    operations_.push_back(e);
  }
}
