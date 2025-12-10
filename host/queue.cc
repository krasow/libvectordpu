#include "queue.h"

#include <thread>

#include "logger.h"
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
  logger.lock() << "[Event(" << me->id << ") "
                << operationtype_to_string(me->op) << "] Callback finished"
                << std::endl;
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

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[Event(" << this->id << ") "
                << operationtype_to_string(this->op) << "] Callback Registered"
                << std::endl;
#endif
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
  logger.lock() << "[EventQueue] Processing id " << e->id << ": "
                << operationtype_to_string(e->op) << " event." << std::endl;
#endif

  switch (e->op) {
    case Event::OperationType::FENCE:
      this->add_fence(e);
      break;
    case Event::OperationType::COMPUTE:
      e->started = true;
      e->cb();
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

    while (!temp_queue.empty()) {
      auto e = temp_queue.front();  // Get the front element
      logger.lock() << "  Event id: " << e->id
                    << ", type: " << operationtype_to_string(e->op)
                    << ", started: " << e->started
                    << ", finished: " << e->finished << std::endl;
      temp_queue.pop_front();  // Pop the element from the temporary queue
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

      for (const auto& e : events) {
        logger.lock() << "  Event id: " << e->id
                      << ", type: " << operationtype_to_string(e->op)
                      << ", started: " << e->started
                      << ", finished: " << e->finished << std::endl;
      }
    } else {
      logger.lock() << "[EventQueue] No active events." << std::endl;
    }
  }
#endif
}