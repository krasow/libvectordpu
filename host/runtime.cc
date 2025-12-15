#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include <thread>

#include "runtime.h"

allocator& DpuRuntime::get_allocator() { return *allocator_; }
EventQueue& DpuRuntime::get_event_queue() { return *event_queue_; }
Logger& DpuRuntime::get_logger() { return *logger_; }
dpu_set_t& DpuRuntime::dpu_set() { return *dpu_set_; }
uint32_t DpuRuntime::num_dpus() const { return num_dpus_; }
uint32_t DpuRuntime::num_tasklets() const { return NR_TASKLETS; }

void DpuRuntime::init(uint32_t num_dpus) {
  if (initialized_) return;  // idempotent
  num_dpus_ = num_dpus;
  logger_ = std::make_unique<Logger>();

#if ENABLE_DPU_LOGGING == 1
  logger_->lock() << "[runtime] Initializing DPU runtime with " << num_dpus_
                  << " DPUs..." << std::endl;
#endif

  // Allocate DPU set
  dpu_set_ = new dpu_set_t();

  std::string backend_str = "backend=";
  backend_str += BACKEND;

  DPU_ASSERT(dpu_alloc(num_dpus_, backend_str.c_str(), dpu_set_));
  DPU_ASSERT(dpu_load(*dpu_set_, DPU_RUNTIME, nullptr));

#if ENABLE_DPU_LOGGING == 1
  logger_->lock() << "[runtime] DPU runtime initialized with " << backend_str
                  << std::endl;
#endif

  // Allocate allocator and event queue
  size_t dpu_mem = 64 * 1024 * 1024;  // 64MB per DPU
  allocator_ = std::make_unique<allocator>(0, dpu_mem, num_dpus_);
  event_queue_ = std::make_unique<EventQueue>();

  initialized_ = true;
}

void DpuRuntime::shutdown() {
  if (!initialized_) return;

#if ENABLE_DPU_LOGGING == 1
  logger_->lock() << "[runtime] Shutting down DPU runtime..." << std::endl;
#endif

  while (event_queue_->has_pending()) {
    logger_->lock() << "[runtime] Waiting for pending events to complete..."
                    << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // if (initialized_) {
  //   DPU_ASSERT(dpu_free(dpu_set_));
  // }

  allocator_.reset();
  event_queue_.reset();
  logger_.reset();
  dpu_set_ = nullptr;

  initialized_ = false;
}
