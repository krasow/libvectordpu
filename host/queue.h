#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <typeindex>
#include <variant>

#include "common.h"
#include "vectordesc.h"

class Event {
 public:
  enum class OperationType { COMPUTE, DPU_TRANSFER, HOST_TRANSFER, FENCE };

  OperationType op;
  std::function<void()> cb;

  std::variant<std::monostate, detail::VectorDescRef> res;

  Event(OperationType t) : op(t), res(std::monostate()) {}

  template <typename Callable>
  Event(OperationType t, Callable&& c)
      : op(t), cb(std::forward<Callable>(c)), res(std::monostate()) {}

  size_t id = 0;
  bool finished = false;
  bool started = false;
  // bool has_parents = false;
  // std::list<std::shared_ptr<Event>> parents;

  void add_completion_callback();
  void mark_finished() { this->finished = true; }
  bool operator==(const Event& other) const { return this->id == other.id; }
};

class EventQueue {
 public:
  EventQueue() = default;
  ~EventQueue() = default;

  void submit(std::shared_ptr<Event> e) {
    e->id = counter_++;
    operations_.push_back(e);
  }

  void add_fence(std::shared_ptr<Event> e);

  void wait();
  bool process_next();
  void process_events(size_t wait_for_id);
  void debug_print_queue();
  void debug_active_events();

  bool has_pending() const { return !operations_.empty(); }
  std::size_t pending_count() const { return operations_.size(); }

  std::shared_ptr<Event> get_curr_event() const { return current_event_; }
  size_t get_curr_event_id() const {
    if (current_event_ != nullptr) {
      return current_event_->id;
    } else {
      return SIZE_MAX;
    }
  }

  std::deque<std::shared_ptr<Event>>::iterator begin() {
    return operations_.begin();
  }
  std::deque<std::shared_ptr<Event>>::iterator end() {
    return operations_.end();
  }
  size_t size() const { return operations_.size(); }

  std::list<std::shared_ptr<Event>>& get_active_events() {
    return running_events_;
  }

  std::mutex& get_mutex() { return mtx_; }

 private:
  std::mutex mtx_;
  size_t counter_ = 1;
  std::shared_ptr<Event> current_event_ = nullptr;
  std::deque<std::shared_ptr<Event>> operations_;
  std::list<std::shared_ptr<Event>> running_events_;
};
