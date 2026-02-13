#include "queue.h"

#include <cassert>
#include <mutex>
#include <ostream>
#include <thread>

#include "runtime.h"
#include "vectordpu.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

std::string operationtype_to_string(Event::OperationType op) {
  switch (op) {
    case Event::OperationType::COMPUTE:
      return "COMPUTE";
    case Event::OperationType::DPU_TRANSFER:
      return "DPU_TRANSFER";
    case Event::OperationType::HOST_TRANSFER:
      return "HOST_TRANSFER";
    case Event::OperationType::FENCE:
      return "FENCE";
    default:
      return "UNKNOWN";
  }
}

/*static*/ dpu_error_t upmem_callback([[maybe_unused]] struct dpu_set_t stream,
                                      [[maybe_unused]] uint32_t rank_id,
                                      void* data) {
  auto self_ptr = static_cast<std::shared_ptr<Event>*>(data);
  std::shared_ptr<Event> me = *self_ptr;

  me->mark_finished(/* true */);

  auto& runtime = DpuRuntime::get();
  auto& events = runtime.get_event_queue().get_active_events();
  std::mutex& events_mutex = runtime.get_event_queue().get_mutex();
  {
    std::lock_guard<std::mutex> lock(events_mutex);
    events.remove(me);
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[event-logger] id=" << me->id
                << " type=" << operationtype_to_string(me->op)
                << " phase=finished" << std::endl;
#endif

  return DPU_OK;
}

void Event::add_completion_callback() {
  assert(this->finished == false);

  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();

  auto wrapper = new std::shared_ptr<Event>(this);

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

void EventQueue::add_fence(std::shared_ptr<Event> e) {
  assert(e->finished == false);

  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();

  auto wrapper = new std::shared_ptr<Event>(std::move(e));

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

bool EventQueue::process_next() {
  if (operations_.empty()) {
    return false;
  }
  std::shared_ptr<Event> e = operations_.front();
  operations_.pop_front();

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
#endif

#if ENABLE_DPU_LOGGING >= 1
  logger.lock() << "[event-logger] id=" << e->id
                << " type=" << operationtype_to_string(e->op)
                << " phase=started" << std::endl;
#endif

  switch (e->op) {
    case Event::OperationType::FENCE:
      this->add_fence(e);
      break;
    case Event::OperationType::COMPUTE:
      e->started = true;
#if PIPELINE
      if (!e->rpn_ops.empty()) {
        // Automatic fusion or manual RPN pipeline
        detail::internal_launch_universal_pipeline(
            e->output, (e->inputs.empty() ? nullptr : e->inputs[0]), e->rpn_ops,
            (e->inputs.size() > 1 ? std::vector<detail::VectorDescRef>(
                                        e->inputs.begin() + 1, e->inputs.end())
                                  : std::vector<detail::VectorDescRef>()),
            e->kid);
      } else if (e->cb) {
        e->cb();
      }
#else
      if (e->cb) {
        e->cb();
      }
#endif
      e->add_completion_callback();
      break;
    case Event::OperationType::DPU_TRANSFER:
      e->started = true;
      e->cb();
      e->add_completion_callback();
      break;
    case Event::OperationType::HOST_TRANSFER:
      e->started = true;
      e->cb();
      e->add_completion_callback();
      break;
    default:
      assert(false && "Unknown event type");
  }

  current_event_ = e;
  running_events_.push_back(e);

  debug_active_events();
  debug_print_queue();
  return true;
}

void EventQueue::process_events(size_t wait_for_id) {
  while (true) {
    bool made_progress = this->process_next();
    if (this->get_curr_event_id() > wait_for_id) {
      break;
    }
    // check if wait_for_id event has completed
    if (this->get_curr_event_id() == wait_for_id &&
        this->get_curr_event() != nullptr && this->get_curr_event()->finished) {
      break;
    }

    // exit if no more events to process
    if (operations_.empty() && running_events_.empty()) break;

    if (!made_progress)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::mutex> lock(mtx_);
  e->id = counter_++;

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
        if (last->rpn_ops.size() + e->rpn_ops.size() <= MAX_PIPELINE_OPS &&
            last->inputs.size() + e->inputs.size() <= MAX_PIPELINE_OPERANDS) {
          // Fusion!
          // last sequence: [ ... ] -> result on stack top
          // e sequence: [ PUSH_INPUT, ... ] where INPUT is last->output

          // Rewrite 'e' sequence to use result on stack instead of pushing it
          std::vector<uint8_t> new_rpn;

          // Promote 'last' if it's not already RPN
          if (last->rpn_ops.empty()) {
            new_rpn.push_back(OP_PUSH_INPUT);
            if (last->inputs.size() > 1) {  // Binary
              new_rpn.push_back(OP_PUSH_OPERAND_0);
            }
            new_rpn.push_back(last->opcode);
            last->kid = last->pipeline_kid;
          } else {
            new_rpn = last->rpn_ops;
          }

          // Promote 'e' to RPN if needed to merge
          std::vector<uint8_t> e_rpn = e->rpn_ops;
          if (e_rpn.empty()) {
            e_rpn.push_back(OP_PUSH_INPUT);
            if (e->inputs.size() > 1) {  // Binary
              e_rpn.push_back(OP_PUSH_OPERAND_0);
            }
            e_rpn.push_back(e->opcode);
          }

          std::vector<detail::VectorDescRef> combined_inputs = last->inputs;

          for (uint8_t op : e_rpn) {
            if (op == OP_PUSH_INPUT) {  // OP_PUSH_INPUT
              if (e->inputs[0] == last->output) {
                // already on stack! do nothing
              } else {
                // Need to push it. Map it to next operand slot
                uint8_t slot = combined_inputs.size() - 1;
                combined_inputs.push_back(e->inputs[0]);
                new_rpn.push_back(OP_PUSH_OPERAND_0 + slot);
              }
            } else if (op >= OP_PUSH_OPERAND_0 &&
                       op <= OP_PUSH_OPERAND_7) {  // OP_PUSH_OPERAND_X
              uint32_t orig_idx = op - OP_PUSH_OPERAND_0;
              if (e->inputs[orig_idx + 1] == last->output) {
                // fused input. already on stack
              } else {
                uint8_t slot = combined_inputs.size() - 1;
                combined_inputs.push_back(e->inputs[orig_idx + 1]);
                new_rpn.push_back(OP_PUSH_OPERAND_0 + slot);
              }
            } else {
              new_rpn.push_back(op);
            }
          }

          last->rpn_ops = new_rpn;
          last->inputs = combined_inputs;
          last->output = e->output;
          fused = true;

#if ENABLE_DPU_LOGGING >= 1
          Logger& logger = DpuRuntime::get().get_logger();
          logger.lock() << "[queue-fuse] fused event id=" << e->id
                        << " into last=" << last->id
                        << " new_ops_count=" << last->rpn_ops.size()
                        << std::endl;
#endif
        }
      }
    }
  }
#endif

  if (!fused) {
    operations_.push_back(e);
  }
}
