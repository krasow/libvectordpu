#pragma once

#include <memory>

#include "allocator.h"
#include "logger.h"
#include "queue.h"

struct dpu_set_t;

class DpuRuntime {
 private:
  DpuRuntime() : initialized_(false) {}
  ~DpuRuntime() = default;

  bool initialized_;
  dpu_set_t* dpu_set_;
  uint32_t num_dpus_;
  std::unique_ptr<allocator> allocator_;
  std::unique_ptr<EventQueue> event_queue_;
  std::unique_ptr<Logger> logger_;

 public:
  // Delete copy/move
  DpuRuntime(const DpuRuntime&) = delete;
  DpuRuntime& operator=(const DpuRuntime&) = delete;

  // Get singleton instance
  static DpuRuntime& get() {
    static DpuRuntime instance;
    return instance;
  }

  void init(uint32_t num_dpus);
  bool is_initialized() const { return initialized_; }

  allocator& get_allocator();
  EventQueue& get_event_queue();
  Logger& get_logger();
  dpu_set_t& dpu_set();
  uint32_t num_dpus() const;
  uint32_t num_tasklets() const;

  void shutdown();
};
