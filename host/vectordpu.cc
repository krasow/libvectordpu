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

#if PIPELINE
// Redefine signatures to include KernelID
void internal_launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype =
        static_cast<uint8_t>(KERNEL_UNARY);  // Universal considered unary-ish?
                                             // Or create KERNEL_PIPELINE?
    // Using KERNEL_UNARY for now as it's just a category

    args[i].num_elements = init->desc[i].size_bytes /
                           init->element_size;  // Assume init defines size
    args[i].size_type = init->element_size;

    args[i].pipeline.init_offset = init->desc[i].ptr;
    args[i].pipeline.res_offset = res->desc[i].ptr;
    args[i].pipeline.num_ops =
        std::min((size_t)ops.size(), (size_t)MAX_PIPELINE_OPS);

    for (size_t j = 0; j < args[i].pipeline.num_ops; ++j) {
      args[i].pipeline.ops[j] = ops[j];
    }

    // Map operands by index (0..MAX_PIPELINE_OPERANDS-1)
    for (size_t j = 0; j < MAX_PIPELINE_OPERANDS; ++j) {
      if (j < operands.size()) {
        args[i].pipeline.binary_operands[j] = operands[j]->desc[i].ptr;
      } else {
        args[i].pipeline.binary_operands[j] = 0;
      }
    }
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
#endif

void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                  uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  (void)pipeline_kid;

  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_unary, res, rhs, kernel_id));

  e->inputs = {rhs};
  e->output = res;
  e->kid = kernel_id;
  e->pipeline_kid = pipeline_kid;
  e->opcode = opcode;
  e->res = res;

  event_queue.submit(e);
#else
  (void)opcode;
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_unary, res, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  e->res = res;
  event_queue.submit(e);
#endif

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE (unary) kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

#if PIPELINE
void launch_universal_pipeline(VectorDescRef res, VectorDescRef init,
                               const std::vector<uint8_t>& ops,
                               const std::vector<VectorDescRef>& operands,
                               KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE);

  e->inputs = {init};
  e->inputs.insert(e->inputs.end(), operands.begin(), operands.end());
  e->output = res;
  e->rpn_ops = ops;
  e->kid = kernel_id;
  e->res = res;

  // Detect reduction and flag result descriptor synchronously
  for (uint8_t op : ops) {
    if (op >= OP_MIN && op <= OP_PRODUCT) {
      res->is_reduction_result = true;
      res->reduction_rid = static_cast<KernelID>(op);
    }
  }

  event_queue.submit(e);
}
#endif

void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id, uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  (void)pipeline_kid;
  assert(lhs->num_elements == rhs->num_elements);

  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_binary, res, lhs, rhs, kernel_id));

  e->inputs = {lhs, rhs};
  e->output = res;
  e->kid = kernel_id;
  e->pipeline_kid = pipeline_kid;
  e->opcode = opcode;
  e->res = res;

  event_queue.submit(e);
#else
  (void)opcode;
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_binary, res, lhs, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  e->res = res;
  event_queue.submit(e);
#endif

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE (binary) kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

void launch_reduction(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                      uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  (void)pipeline_kid;

  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_reduction, res, rhs, kernel_id));

  e->inputs = {rhs};
  e->output = res;
  e->kid = kernel_id;
  e->pipeline_kid = pipeline_kid;
  e->opcode = opcode;
  e->res = res;

  // Mark result description as reduction synchronously
  res->is_reduction_result = true;
  res->reduction_rid = static_cast<KernelID>(opcode);

  event_queue.submit(e);
#else
  (void)opcode;
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_reduction, res, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  e->res = res;
  event_queue.submit(e);
#endif

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE (reduction) kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

}  // namespace detail
