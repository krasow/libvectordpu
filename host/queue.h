#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <typeindex>
#include <variant>

#include "common.h"
#include "vectordesc.h"

class Event;
class EventQueue;

class Event : public std::enable_shared_from_this<Event> {
 public:
  enum class OperationType { COMPUTE, DPU_TRANSFER, HOST_TRANSFER, FENCE };

  OperationType op;
  std::function<void()> cb;

  // Metadata for fusion
  std::vector<detail::VectorDescRef> inputs;
  detail::VectorDescRef output;
  std::vector<uint8_t> rpn_ops;
  std::vector<uint32_t> scalars;
  KernelID kid = 0;
  KernelID pipeline_kid = 0;  // For lazy promotion
  uint8_t opcode = 0;         // For lazy promotion
  bool is_scalar = false;     // Prevent fusion of scalar ops for now
  uint32_t scalar_value = 0;  // Scalar operand for fusion
  void* host_ptr = nullptr;   // For transfers
  size_t transfer_size = 0;   // For transfers

  // JIT
  std::string jit_binary_path;
  bool is_locked_for_jit = false;
  int jit_sub_kernel_idx = -1;
  std::shared_future<std::string> jit_future;

  Event(OperationType t) : op(t) {}

  template <typename Callable>
  Event(OperationType t, Callable&& c) : op(t), cb(std::forward<Callable>(c)) {}

  size_t id = 0;
  std::string slice_name;
  std::set<size_t> dependencies;
  std::atomic<bool> finished{false};
  bool started = false;
  size_t max_id = 0;  // The highest ID represented by this event (if fused)

  void add_completion_callback(std::shared_ptr<Event> self);
  void mark_finished() { this->finished.store(true); }
  bool operator==(const Event& other) const { return this->id == other.id; }
};

struct CallbackData {
  std::shared_ptr<Event> event;
  EventQueue* queue;
};

class EventQueue {
 public:
  static constexpr size_t DEFAULT_MAX_QUEUE_DEPTH = 1024;

  EventQueue() = default;
  ~EventQueue() = default;

  bool try_fuse(std::shared_ptr<Event> last, std::shared_ptr<Event> e);
  void lock_for_jit(std::shared_ptr<Event> e);
  void flush_jit_batch();

  void submit(std::shared_ptr<Event> e);
  void set_max_queue_depth(size_t depth) { max_queue_depth_ = depth; }
  size_t max_queue_depth() const { return max_queue_depth_; }

  void add_fence(std::shared_ptr<Event> e);

  bool process_next();
  void process_events(size_t wait_for_id);
  void wait_running_events();
  void debug_print_queue();
  void debug_active_events();
  size_t count_internal_references(detail::VectorDescRef vec);

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

  size_t get_last_finished_id() const { return last_finished_id_.load(); }

  std::list<std::shared_ptr<Event>>& get_active_events() {
    return running_events_;
  }

  std::mutex& get_mutex() { return mtx_; }

  std::atomic<size_t> last_finished_id_{0};
  std::atomic<int> outstanding_callbacks_{0};
  std::atomic<bool> oom_detected_{false};

 private:
  std::mutex mtx_;
  size_t counter_ = 1;
  size_t max_queue_depth_ = DEFAULT_MAX_QUEUE_DEPTH;
  std::shared_ptr<Event> current_event_ = nullptr;
  std::deque<std::shared_ptr<Event>> operations_;
  std::list<std::shared_ptr<Event>> running_events_;

  // JIT Batching State
  std::vector<std::tuple<std::vector<uint8_t>, std::string, std::string>>
      pending_unique_kernels_;
  std::vector<std::shared_ptr<Event>> pending_jit_events_;
  std::shared_future<std::string> latest_jit_future_;
  std::string current_binary_path_;
};
