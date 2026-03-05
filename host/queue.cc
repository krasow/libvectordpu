#include "queue.h"

#include <cassert>
#include <mutex>
#include <ostream>
#include <thread>

#include "jit.h"
#include "opinfo.h"
#include "perfetto/trace.h"
#include "perfetto/trace_internal.h"
#include "runtime.h"
#include "vectordpu.h"

#if PIPELINE
static uint8_t map_to_var_op(uint8_t op) {
  switch (op) {
    case OP_ADD_SCALAR: return OP_ADD_SCALAR_VAR;
    case OP_SUB_SCALAR: return OP_SUB_SCALAR_VAR;
    case OP_MUL_SCALAR: return OP_MUL_SCALAR_VAR;
    case OP_DIV_SCALAR: return OP_DIV_SCALAR_VAR;
    case OP_ASR_SCALAR: return OP_ASR_SCALAR_VAR;
    default: return op;
  }
}
#endif

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#define MAX_RETRIES 100000

/*static*/ dpu_error_t upmem_callback([[maybe_unused]] struct dpu_set_t stream,
                                      [[maybe_unused]] uint32_t rank_id,
                                      void* data) {
  auto* cb_data = static_cast<CallbackData*>(data);
  std::shared_ptr<Event> me = cb_data->event;
  EventQueue& queue = *(cb_data->queue);

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

    static std::atomic<size_t> callback_count{0};
    size_t count = ++callback_count;
    if (count % 100 == 0) {
#if ENABLE_DPU_LOGGING >= 1
      // Safe to access singleton here because outstanding_callbacks_ > 0
      // and we hold the lock, so shutdown() is blocked.
      Logger& logger = DpuRuntime::get().get_logger();
      logger.lock() << "[queue-heartbeat] callback fired (" << count
                    << ") for id=" << me->id << std::endl;
#endif
    }

    queue.outstanding_callbacks_--;
  }

  delete cb_data;

  return DPU_OK;
}

void Event::add_completion_callback(std::shared_ptr<Event> self) {
  assert(this->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto* cb_data = new CallbackData{self, &queue};

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)cb_data,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

void EventQueue::add_fence(std::shared_ptr<Event> e) {
  assert(e->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto* cb_data = new CallbackData{std::move(e), &queue};

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)cb_data,
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
#if ENABLE_DPU_LOGGING >= 1
    size_t loop_count = 0;
#endif
    size_t retries = 0;
    while (this->get_last_finished_id() < max_dep) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (++retries > MAX_RETRIES) {
          fprintf(stderr, "[debug] Timeout at line 166 (dependency wait)\n");
          throw DpuTimeoutException();
      }
#if ENABLE_DPU_LOGGING >= 1
      if (retries % 1000 == 0) {
        logger.lock() << "[queue-heartbeat] id=" << e->id
                      << " dependency block on " << max_dep
                      << " (current=" << this->get_last_finished_id() << ")"
                      << std::endl;
      }
#endif
    }
  }

  // 1. Initial Naming (Base)
  if (e->slice_name.empty()) {
    e->slice_name = operationtype_to_string(e->op);
  }

  // 2. Fusion and JIT Locking
  {
    std::lock_guard<std::mutex> lock(mtx_);

    // Look-ahead fusion
#if PIPELINE
    if (e->op == Event::OperationType::COMPUTE) {
      size_t lookahead_count = 0;
      while (lookahead_count < MAX_FUSION_LOOKAHEAD_LENGTH &&
             !operations_.empty()) {
        auto next = operations_.front();
        if (next->op != Event::OperationType::COMPUTE) break;

        if (try_fuse(e, next)) {
          operations_.pop_front();
          lookahead_count++;
#if ENABLE_DPU_LOGGING >= 1
          logger.lock() << "[queue-fuse] Look-ahead fused event id=" << next->id
                        << " into id=" << e->id << std::endl;
#endif
        } else {
          break;
        }

        if (operations_.empty()) {
          std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
      }
    }
#endif

#if JIT
    if (e->op == Event::OperationType::COMPUTE && MAX_JIT_QUEUE_DEPTH > 0) {
      if (!e->is_locked_for_jit) {
        lock_for_jit(e);
      }

      if (!e->jit_future.valid()) {
        if (operations_.empty()) {
          std::this_thread::sleep_for(std::chrono::microseconds(1000));
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

        if (!e->jit_future.valid()) {
          flush_jit_batch();
        }
      }
    }
#endif
  }

  // 3. JIT Wait (Outside mutex)
#if JIT
  if (e->op == Event::OperationType::COMPUTE && e->jit_future.valid()) {
#if ENABLE_DPU_LOGGING >= 1
    logger.lock() << "[queue-jit] Awaiting background JIT compilation for id="
                  << e->id << std::endl;
#endif
    if (latest_jit_future_.valid()) {
      e->jit_binary_path = latest_jit_future_.get();
    } else {
      e->jit_binary_path = e->jit_future.get();
    }
  }
#endif

  // 4. Refined Naming
  if (e->op == Event::OperationType::COMPUTE) {
    if (!e->rpn_ops.empty()) {
      std::string ops_list;
      for (size_t i = 0; i < e->rpn_ops.size(); ++i) {
        uint8_t op = e->rpn_ops[i];
        std::string s = opcode_to_string(op);
        if (s.empty()) continue;
        if (!ops_list.empty()) ops_list += ", ";
        ops_list += s;
        if (IS_OP_SCALAR(op)) i += sizeof(uint32_t);
        else if (IS_OP_SCALAR_VAR(op)) i += 1;
      }
      e->slice_name =
          e->rpn_ops.size() > 2 ? "Fused: [" + ops_list + "]" : ops_list;
    } else {
      e->slice_name = kernel_id_to_string(e->kid);
    }

    if (!e->jit_binary_path.empty()) {
      e->slice_name += " (from " + e->jit_binary_path + ")";
    }
  }

#if ENABLE_DPU_LOGGING >= 1
  logger.lock() << "[queue-exec] id=" << e->id << " name=\"" << e->slice_name
                << "\"" << std::endl;
#endif

  // 5. Register with running events and start trace
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (running_events_.empty()) {
      trace::execution_begin(e);
    }
    running_events_.push_back(e);
    current_event_ = e;
    trace::active_ops_counter(running_events_.size());
  }

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
      // Explicitly poll the DPU driver to pump callbacks
      dpu_set_t& dpu_set = DpuRuntime::get().dpu_set();
      dpu_sync(dpu_set);
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
              dynamic_kid, e->scalars);
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
    static int consecutive_ooms = 0;
    consecutive_ooms++;
    if (consecutive_ooms % 100 == 0) {
        fprintf(stderr, "[vectordpu] WARNING: DPU OOM detected for event id=%zu. System is busy-waiting for MRAM to be freed. If using a managed language (e.g. Julia), ensure Garbage Collection is firing.\n", e->id);
    }

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
    oom_detected_.store(true);
    // no progress as we caught an exception
    return NO_PROGRESS;
  }

  // If we made it here, no OOM happened this time
  // But we need to reset the consecutive_ooms counter. 
  // This is tricky without a member variable. 
  // Let's just keep it simple and focused on the warning.

  debug_active_events();
  debug_print_queue();
  return YES_PROGRESS;
}

void EventQueue::process_events(size_t wait_for_id) {
  size_t retries = 0;
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
      bool running = false;
      {
          std::lock_guard<std::mutex> lock(mtx_);
          running = !running_events_.empty();
      }
      if (oom_detected_.load()) {
          if (running) {
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
          } else {
              oom_detected_.store(false);
              throw DpuOOMException();
          }
      } else {
          // Explicitly poll the DPU driver to pump callbacks
          dpu_set_t& dpu_set = DpuRuntime::get().dpu_set();
          dpu_sync(dpu_set);
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          if (++retries > MAX_RETRIES) {
              if (running) {
                  retries = 0;
              } else {
                  fprintf(stderr, "[debug] Timeout at line 468 (event queue process)\n");
                  throw DpuTimeoutException();
              }
          }
      }
    } else {
        retries = 0;
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

void EventQueue::wait_running_events() {
  size_t retries = 0;
  while (true) {
    bool any_running = false;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      any_running = !running_events_.empty();
    }
    if (!any_running) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (++retries > MAX_RETRIES) {
        fprintf(stderr, "[debug] Timeout at line 503 (wait_running_events)\n");
        throw DpuTimeoutException();
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

  latest_jit_future_ = future;

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
      e->rpn_ops.push_back(map_to_var_op(e->opcode));
      e->rpn_ops.push_back(0); // placeholder index
      e->scalars.push_back(e->scalar_value);
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
    if (tn == "i" || tn == "int" || tn == "int32_t")
      type_name = "int32_t";
    else if (tn == "j" || tn == "uint32_t")
      type_name = "uint32_t";
    else if (tn == "f" || tn == "float")
      type_name = "float";
    else if (tn == "d" || tn == "double")
      type_name = "double";
    else if (tn == "l" || tn == "int64_t")
      type_name = "int64_t";
    else if (tn == "m" || tn == "uint64_t")
      type_name = "uint64_t";
    else if (tn == "x" || tn == "long long")
      type_name = "int64_t";
    else if (tn == "y" || tn == "unsigned long long")
      type_name = "uint64_t";
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
    // Flush if we accumulated enough total events
    if (pending_jit_events_.size() >= MAX_JIT_QUEUE_DEPTH) {
      flush_jit_batch();
    }
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

  if (pending_jit_events_.size() >= MAX_JIT_QUEUE_DEPTH) {
    flush_jit_batch();
  }

  // 5. No automatic flush here to allow batching.
}
#endif

bool EventQueue::try_fuse(std::shared_ptr<Event> last,
                          std::shared_ptr<Event> e) {
#if PIPELINE
  if (last->op != Event::OperationType::COMPUTE ||
      e->op != Event::OperationType::COMPUTE || last->output == nullptr) {
    return false;
  }

  // Prevent fusion if already locked for JIT (signature is fixed)
  if (last->is_locked_for_jit || e->is_locked_for_jit) {
    return false;
  }

  // Prevent fusion if last is a reduction (must be the final op)
  if (!last->rpn_ops.empty() && IS_OP_REDUCTION(last->rpn_ops.back())) {
    return false;
  }

  // Look for dependency: does any input of 'e' match 'last->output'?
  bool dependent = false;
  for (size_t i = 0; i < e->inputs.size(); ++i) {
    if (e->inputs[i] == last->output) {
      dependent = true;
      break;
    }
  }

  if (!dependent) return false;

  // Safety check: if last->output is used by more than just (last, e),
  // we must materialize it to MRAM for other consumers.
  // shared_ptr use_count:
  // 1: last->output
  // 2: whoever else (user code, or another event in the queue)
  // Each matching input in 'e' also holds a reference.
  // So we count how many times 'last->output' appears in 'e->inputs'.
  long internal_refs = 1;  // last->output
  for (const auto& in : e->inputs) {
    if (in == last->output) internal_refs++;
  }

  // Count references within the whole library's internal state
  long lib_refs = count_internal_references(last->output);
  // Add reference from 'e' which is not yet in the queue
  for (const auto& in : e->inputs) {
    if (in == last->output) lib_refs++;
  }

  // Safety check: if lib_refs > internal_refs, it means SOME OTHER event in the
  // queue or JIT pending list also needs this vector as input/output.
  if (lib_refs > internal_refs) {
    return false;
  }

  if (last->output.use_count() > (long)lib_refs * 2 + 3) {
    return false;
  }
  // 1. Total ops <= MAX_PIPELINE_OPS
  // 2. Can we map inputs? (max MAX_PIPELINE_OPERANDS binary operands)

  // Promote 'last' RPN if needed to estimate new size
  std::vector<uint8_t> last_rpn = last->rpn_ops;
  if (last_rpn.empty()) {
    if (last->is_scalar) {
      last_rpn.push_back(OP_PUSH_INPUT);
      last_rpn.push_back(map_to_var_op(last->opcode));
      last_rpn.push_back(0); // placeholder index
    } else {
      last_rpn.push_back(OP_PUSH_INPUT);
      if (last->inputs.size() > 1) last_rpn.push_back(OP_PUSH_OPERAND_0);
      last_rpn.push_back(last->opcode);
    }
  }

  // Promote 'e' RPN if needed
  std::vector<uint8_t> e_rpn = e->rpn_ops;
  if (e_rpn.empty()) {
    if (e->is_scalar) {
      if (!e->inputs.empty()) e_rpn.push_back(OP_PUSH_INPUT);
      e_rpn.push_back(map_to_var_op(e->opcode));
      e_rpn.push_back(0); // placeholder index
    } else {
      if (!e->inputs.empty()) e_rpn.push_back(OP_PUSH_INPUT);
      if (e->inputs.size() > 1) e_rpn.push_back(OP_PUSH_OPERAND_0);
      e_rpn.push_back(e->opcode);
    }
  }

  std::vector<detail::VectorDescRef> combined_inputs = last->inputs;
  auto get_operand_push_op = [&](detail::VectorDescRef vec) -> uint8_t {
    if (vec == last->output) return 0;  // Already on stack
    if (combined_inputs.empty()) {
      combined_inputs.push_back(vec);
      return OP_PUSH_INPUT;
    }
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
  size_t last_scalars_size = last->rpn_ops.empty() && last->is_scalar ? 1 : last->scalars.size();
  for (size_t k = 0; k < e_rpn.size(); ++k) {
    uint8_t op = e_rpn[k];
    if (IS_OP_SCALAR(op)) {
      e_rpn_mapped.push_back(op);
      for (int m = 0; m < 4; ++m) {
        if (++k < e_rpn.size()) e_rpn_mapped.push_back(e_rpn[k]);
      }
    } else if (IS_OP_SCALAR_VAR(op)) {
      e_rpn_mapped.push_back(op);
      if (++k < e_rpn.size()) {
        e_rpn_mapped.push_back(last_scalars_size + e_rpn[k]);
      }
    } else if (op == OP_PUSH_INPUT) {
      uint8_t push_op = get_operand_push_op(e->inputs[0]);
      if (push_op == 0xFF) {
        possible = false;
        break;
      }
      if (push_op != 0) e_rpn_mapped.push_back(push_op);
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_PIPELINE_OPERANDS) {
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

  if (possible && (last_rpn.size() + e_rpn_mapped.size() > MAX_PIPELINE_OPS)) {
    possible = false;
  }

  if (possible) {
    if (last->rpn_ops.empty()) {
      last->rpn_ops = last_rpn;
      last->kid = last->pipeline_kid;
      if (last->is_scalar) last->scalars.push_back(last->scalar_value);
    }
    last->rpn_ops.insert(last->rpn_ops.end(), e_rpn_mapped.begin(),
                         e_rpn_mapped.end());
    if (e->is_scalar) last->scalars.push_back(e->scalar_value);
    else last->scalars.insert(last->scalars.end(), e->scalars.begin(), e->scalars.end());
    last->inputs = combined_inputs;
    last->output = e->output;
    last->max_id = std::max(last->max_id, e->id);

    // Update dependencies for fused event
    for (const auto& in : e->inputs) {
      if (in && in->last_producer_id != 0 && in->last_producer_id != last->id) {
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
      } else if (IS_OP_SCALAR_VAR(op)) {
        i += 1;
      }
    }
    last->slice_name = "Fused: [" + ops_list + "]";

#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[queue-fuse] fused event id=" << e->id
                  << " into last=" << last->id
                  << " new_ops_count=" << last->rpn_ops.size() << std::endl;
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
      } else if (IS_OP_SCALAR_VAR(op)) {
        i += 1;
      }
    }

    trace::event_fused(e, last, fused_ops);
    trace::inqueue_end(e);
    return true;
  }
#endif
  return false;
}

size_t EventQueue::count_internal_references(detail::VectorDescRef vec) {
  if (!vec) return 0;
  size_t count = 0;
  auto count_in_event = [&](std::shared_ptr<Event> ev) {
    if (!ev) return;
    if (ev->output == vec) count++;
    for (const auto& in : ev->inputs) {
      if (in == vec) count++;
    }
  };

  if (current_event_) count_in_event(current_event_);
  for (auto& ev : operations_) count_in_event(ev);
  for (auto& ev : running_events_) count_in_event(ev);
  for (auto& ev : pending_jit_events_) count_in_event(ev);

  return count;
}

void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::mutex> lock(mtx_);
  // Backpressure: block if total in-flight events (pending + running) exceed
  // limit. Running events hold shared_ptr<VectorDesc> references that prevent
  // DPU MRAM deallocation, so we must count them too—not just the pending
  // operations_ queue.
  while (operations_.size() + running_events_.size() >= max_queue_depth_) {
    if (oom_detected_.load()) {
        oom_detected_.store(false);
        throw DpuOOMException();
    }
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
    if (try_fuse(last, e)) {
      fused = true;
    }
  }
#endif

  if (!fused) {
#if JIT
    // If this is a pipeline breaker (reduction or non-compute), lock everything
    if (e->op != Event::OperationType::COMPUTE ||
        (IS_OP_REDUCTION(e->opcode) && !e->is_scalar)) {
      bool any_locked = false;
      for (auto& op : operations_) {
        if (op->op == Event::OperationType::COMPUTE && !op->is_locked_for_jit) {
          lock_for_jit(op);
          any_locked = true;
        }
      }
      if (any_locked) {
        flush_jit_batch();
      }
    }
    // Automatically promote all single compute events to batched kernels
    if (e->op == Event::OperationType::COMPUTE && e->rpn_ops.empty() &&
        MAX_JIT_QUEUE_DEPTH > 0) {
      e->kid = e->pipeline_kid;
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
