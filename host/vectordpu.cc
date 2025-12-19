#include "vectordpu.h"

#include "vectordesc.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

namespace detail {

void vec_xfer_to_dpu(char* cpu, VectorDescRef desc) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu[element])));
    element += desc->desc[idx_dpu].size_bytes - desc->reserved_bytes;
  }

  uint32_t mram_location = desc->desc[0].ptr;
  size_t xfer_size = desc->desc[0].size_bytes - desc->reserved_bytes;
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

void vec_xfer_from_dpu(char* cpu, VectorDescRef desc) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu[element])));
    element += desc->desc[idx_dpu].size_bytes - desc->reserved_bytes;
  }

  uint32_t mram_location = desc->desc[0].ptr;
  size_t xfer_size = desc->desc[0].size_bytes - desc->reserved_bytes;
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

void internal_launch_binary(VectorDescRef res, VectorDescRef lhs,
                            VectorDescRef rhs, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_BINARY);
    args[i].num_elements = rhs->desc[i].size_bytes / rhs->element_size;
    args[i].size_type = rhs->element_size;
    args[i].binary.lhs_offset = (lhs->desc[i].ptr);
    args[i].binary.rhs_offset = (rhs->desc[i].ptr);
    args[i].binary.res_offset = (res->desc[i].ptr);
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  log_dpu_launch_args(logger, args, nr_of_dpus);
#endif

  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;
  uint32_t idx_dpu = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
}

void internal_launch_unary(VectorDescRef res, VectorDescRef rhs,
                           KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_UNARY);
    args[i].num_elements = rhs->desc[i].size_bytes / rhs->element_size;
    args[i].size_type = rhs->element_size;
    args[i].unary.rhs_offset = (rhs->desc[i].ptr);
    args[i].unary.res_offset = (res->desc[i].ptr);
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  log_dpu_launch_args(logger, args, nr_of_dpus);
#endif

  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;
  uint32_t idx_dpu = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
}

void internal_launch_reduction(VectorDescRef res, VectorDescRef rhs,
                               KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  uint32_t nr_of_dpus = runtime.num_dpus();

  DPU_LAUNCH_ARGS args[nr_of_dpus];
  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_REDUCTION);
    args[i].num_elements = rhs->desc[i].size_bytes / rhs->element_size;
    args[i].size_type = rhs->element_size;
    args[i].reduction.rhs_offset = (rhs->desc[i].ptr);
    args[i].reduction.res_offset = (res->desc[i].ptr);
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  log_dpu_launch_args(logger, args, nr_of_dpus);
#endif

  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;
  uint32_t idx_dpu = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
}

void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id) {
  assert(lhs->num_elements == rhs->num_elements);

  auto bound_cb = std::bind(internal_launch_binary, res, lhs, rhs, kernel_id);
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = res;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id) {
  auto bound_cb = std::bind(internal_launch_unary, res, rhs, kernel_id);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = res;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

void launch_reduction(VectorDescRef buf, VectorDescRef rhs,
                      KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  auto bound_cb = std::bind(internal_launch_reduction, buf, rhs, kernel_id);
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = buf;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

}  // namespace detail