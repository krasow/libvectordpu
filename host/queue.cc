#include "queue.h"

#include <cassert>
#include <mutex>
#include <ostream>
#include <thread>

#include "perfetto/trace.h"
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

  TRACE_CALLBACK_THREAD();

  me->mark_finished(/* true */);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  auto& events = queue.get_active_events();
  std::mutex& mtx = queue.get_mutex();

  {
    std::lock_guard<std::mutex> lock(mtx);
    TRACE_EXECUTION_END();
    events.remove(me);
    if (!events.empty()) {
      auto next = events.front();
      TRACE_EXECUTION_BEGIN_NEXT(next);
    }
  }

  TRACE_ACTIVE_OPS(queue.get_active_events().size());

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[event-logger] id=" << me->id
                << " type=" << operationtype_to_string(me->op)
                << " phase=finished" << std::endl;
#endif

  delete self_ptr;

  return DPU_OK;
}

void Event::add_completion_callback(std::shared_ptr<Event> self) {
  assert(this->finished == false);

  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();

  auto wrapper = new std::shared_ptr<Event>(self);

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
  std::shared_ptr<Event> e;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (operations_.empty()) {
      return false;
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

  TRACE_INQUEUE_END(e);

  e->slice_name = operationtype_to_string(e->op);
  if (e->op == Event::OperationType::COMPUTE) {
    e->slice_name = kernel_id_to_string(e->kid);
    if (!e->rpn_ops.empty()) {
      e->slice_name += " (Fused)";
    }
  }

  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (running_events_.empty()) {
      TRACE_EXECUTION_BEGIN(e);
    }
    running_events_.push_back(e);
    current_event_ = e;

    TRACE_ACTIVE_OPS(running_events_.size());
  }

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
        e->inputs.clear();  // Release inputs early
      } else if (e->cb) {
        e->cb();
        e->inputs.clear();  // Release inputs early
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
  // Implement backpressure: block if queue is too full
  const size_t MAX_QUEUE_DEPTH = 128;
  while (operations_.size() >= MAX_QUEUE_DEPTH) {
    mtx_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mtx_.lock();
  }

  e->id = counter_++;

  TRACE_EVENT_ENQUEUED(e, operations_, running_events_);

  bool fused = false;
#if PIPELINE
  if (e->op == Event::OperationType::COMPUTE && !operations_.empty()) {
    auto last = operations_.back();
    if (last->op == Event::OperationType::COMPUTE && last->output != nullptr &&
        !e->is_scalar && !last->is_scalar) {
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
          std::vector<uint8_t> last_rpn = last->rpn_ops;
          if (last_rpn.empty()) {
            last_rpn.push_back(OP_PUSH_INPUT);
            if (last->inputs.size() > 1) last_rpn.push_back(OP_PUSH_OPERAND_0);
            last_rpn.push_back(last->opcode);
          }

          // Promote 'e' RPN if needed
          std::vector<uint8_t> e_rpn = e->rpn_ops;
          if (e_rpn.empty()) {
            e_rpn.push_back(OP_PUSH_INPUT);
            if (e->inputs.size() > 1) e_rpn.push_back(OP_PUSH_OPERAND_0);
            e_rpn.push_back(e->opcode);
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
          for (uint8_t op : e_rpn) {
            if (op == OP_PUSH_INPUT) {
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
            for (uint8_t op : last->rpn_ops) {
              std::string s = opcode_to_string(op);
              if (s.empty()) continue;
              if (!ops_list.empty()) ops_list += ", ";
              ops_list += s;
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
            if (e_rpn_mapped.size() == 1) {
              fused_ops = opcode_to_string(e_rpn_mapped[0]);
            } else {
              for (uint8_t op : e_rpn_mapped) {
                std::string s = opcode_to_string(op);
                if (s.empty()) continue;
                if (!fused_ops.empty()) fused_ops += ", ";
                fused_ops += s;
              }
            }

            TRACE_EVENT_FUSED(e, last, fused_ops);
          }
        }
      }
    }
  }
#endif

  if (!fused) {
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
