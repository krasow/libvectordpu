#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include <string>
#include <thread>

// dl function info
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>

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

  Dl_info info;
  void* fptr = (void*)&DpuRuntime::get;  // static function pointer
  if (dladdr(fptr, &info) == 0) {
    std::__throw_runtime_error("Failed to get library path");
  }
  // Full path to libvectordpu.so
  const char* lib_path = info.dli_fname;
  // Directory containing the library
  std::string lib_dir = dirname(const_cast<char*>(lib_path));
  // Compute path to runtime.dpu relative to the library
  std::string dpu_file = lib_dir + "/../bin/runtime.dpu";

  DPU_ASSERT(dpu_load(*dpu_set_, dpu_file.c_str(), nullptr));

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

void DpuRuntime::debug_read_dpu_log() {
  dpu_set_t dpu;
  dpu_set_t& set = this->dpu_set();
  DPU_FOREACH(set, dpu) { DPU_ASSERT(dpu_log_read(dpu, stdout)); }
}