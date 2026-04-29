#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "perfetto/trace.h"

template <typename T>
dpu_vector<T>::dpu_vector() noexcept : size_(0), reserved_(0) {}

template <typename T>
dpu_vector<T>::dpu_vector(uint32_t n, uint32_t reserved, bool lazy,
                          std::string_view name, std::source_location loc)
    : size_(n),
      reserved_(reserved),
      debug_name(name.data()),
      debug_file(loc.file_name()),
      debug_line(loc.line()) {
  auto& runtime = DpuRuntime::get();
  if (runtime.is_initialized() == false) {
    int nr_dpus = 8;
    const char* env_val = std::getenv("NR_DPUS");
    if (env_val != nullptr) {
      nr_dpus = std::atoi(env_val);
    }
    runtime.init(nr_dpus);
  }
  data_ = runtime.get_allocator().allocate_upmem_vector(n, reserved, sizeof(T),
                                                        lazy);
  data_->type_name = typeid(T).name();
  data_->debug_name = debug_name;
  data_->debug_file = debug_file;
  data_->debug_line = debug_line;
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = runtime.get_logger();
  log_allocation(logger, typeid(T), n, debug_name, debug_file, debug_line);
#endif
}

template <typename T>
dpu_vector<T>::dpu_vector(const dpu_vector& other)
    : data_(other.data_),
      size_(other.size_),
      reserved_(other.reserved_),
      debug_name(other.debug_name),
      debug_file(other.debug_file),
      debug_line(other.debug_line),
      copied(false) {
  other.copied = true;
}

template <typename T>
dpu_vector<T>::dpu_vector(dpu_vector&& other) noexcept
    : data_(std::move(other.data_)),
      size_(other.size_),
      reserved_(other.reserved_),
      debug_name(other.debug_name),
      debug_file(other.debug_file),
      debug_line(other.debug_line),
      copied(false) {
  // ownership handled by shared_ptr
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(const dpu_vector& other) {
  if (this != &other) {
    data_ = other.data_;
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
  }
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(dpu_vector&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
  }
  return *this;
}

template <typename T>
dpu_vector<T>::~dpu_vector() {}

template <typename T>
void dpu_vector<T>::add_fence() {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::FENCE);

  event_queue.submit(e);
  event_queue.process_events(e->id);
}

inline void dpu_fence() { DpuRuntime::get().get_event_queue().sync(); }

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec,
                                      std::string_view name,
                                      std::source_location loc) {
  dpu_vector<T> vec(cpu_vec.size(), 0, false, name, loc);
  auto desc = vec.data_desc_ref();

  char* cpu_buffer = reinterpret_cast<char*>(cpu_vec.data());
  auto bound_cb = std::bind(detail::vec_xfer_to_dpu, cpu_buffer, desc);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::DPU_TRANSFER, bound_cb);
  e->output = desc;
  e->host_ptr = cpu_buffer;
  e->transfer_size = cpu_vec.size() * sizeof(T);

  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=DPU_TRANSFER size=" << cpu_vec.size()
                << std::endl;
#endif
  return vec;
}

template <typename T>
vector<T> dpu_vector<T>::to_cpu() {
  auto desc = this->data_desc_ref();
  // Allocate CPU buffer large enough to hold all data
  size_t total_size = this->size();
  auto& runtime = DpuRuntime::get();
  size_t num_dpus = runtime.num_dpus();
  size_t min_xfer = 8;  // 8 bytes

  // Compute bytes per DPU
  size_t bytes_per_dpu = (total_size * sizeof(T)) / num_dpus;

  // Ensure at least 8 bytes per DPU
  if (total_size == num_dpus && bytes_per_dpu < min_xfer) {
    // Round up to the number of elements that makes 8 bytes per DPU
    size_t elems_per_dpu =
        (min_xfer + sizeof(T) - 1) / sizeof(T);  // ceil(min_xfer /sizeof(T))
    total_size = num_dpus * elems_per_dpu;
  }

  vector<T> cpu_vec(total_size);

  char* cpu_buffer = reinterpret_cast<char*>(cpu_vec.data());
  auto bound_cb = std::bind(detail::vec_xfer_from_dpu, cpu_buffer, desc);
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);
  e->inputs = {desc};
  e->host_ptr = cpu_buffer;
  e->transfer_size = cpu_vec.size() * sizeof(T);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[VECTOR-TO-CPU] submit size=" << cpu_vec.size()
                << " bytes=" << e->transfer_size << " id=" << e->id
                << std::endl;
#endif
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[queue-append] type=HOST_TRANSFER size=" << cpu_vec.size()
                << std::endl;
#endif

// Auto-fence after DPU->HOST transfer if enabled
#if ENABLE_AUTO_FENCING == 1
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[VECTOR-TO-CPU] wait id=" << e->id << std::endl;
#endif
  event_queue.process_events(e->id);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[VECTOR-TO-CPU] done id=" << e->id << std::endl;
#endif
// need the event to be completed before reading printf output
#if ENABLE_DPU_PRINTING == 1
  // read and print DPU logs to host stdout
  runtime.debug_read_dpu_log();
#endif
#endif

  return cpu_vec;
}

template <typename T>
typename dpu_vector<T>::reduction_result_t reduction_cpu(dpu_vector<T>& da,
                                                         KernelID kernel_id) {
  // block and send to cpu
  auto a = da.to_cpu();

  uint64_t flow_id =
      (da.data_desc_ref() ? da.data_desc_ref()->last_producer_id : 0);
  trace::reduction_cpu _trace(flow_id);

  auto& runtime = DpuRuntime::get();
  assert(a.size() % runtime.num_dpus() == 0);
  size_t stride = a.size() / runtime.num_dpus();
  // initialize accumulator with the first partial result
  typename dpu_vector<T>::reduction_result_t acc = a[0];

  // reduce over the remaining DPUs
  auto op = kernel_infos[kernel_id].op;
  for (size_t i = stride; i < a.size(); i += stride) {
    typename dpu_vector<T>::reduction_result_t x = a[i];
    switch (op) {
      case KERNEL_OP_SUM:
        acc += x;
        break;
      case KERNEL_OP_PRODUCT:
        acc *= x;
        break;
      case KERNEL_OP_MAX:
        acc = (x > acc) ? x : acc;
        break;
      case KERNEL_OP_MIN:
        acc = (x < acc) ? x : acc;
        break;
      default:
        assert(false && "Unknown reduction operation");
    }
  }
  return acc;
}

// Binary operators
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::add, OpInfo<T>::add_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::sub, OpInfo<T>::sub_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::mul, OpInfo<T>::mul_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator/(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::div, OpInfo<T>::div_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator+=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::add,
                        OpInfo<T>::add_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator-=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::sub,
                        OpInfo<T>::sub_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator*=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::mul,
                        OpInfo<T>::mul_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator/=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::div,
                        OpInfo<T>::div_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator+=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::add_scalar,
                               OpInfo<T>::add_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator-=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::sub_scalar,
                               OpInfo<T>::sub_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator*=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::mul_scalar,
                               OpInfo<T>::mul_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator/=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::div_scalar,
                               OpInfo<T>::div_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator>>=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::asr_scalar,
                               OpInfo<T>::asr_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::operator-() const {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_unary(res.data_desc_ref(), this->data_desc_ref(),
                       OpInfo<T>::negate, OpInfo<T>::negate_op,
                       OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator>>(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::asr_scalar, OpInfo<T>::asr_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::operator==(T scalar) const {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(res.data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::eq_scalar,
                               OpInfo<T>::eq_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::add_scalar, OpInfo<T>::add_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(T lhs, const dpu_vector<T>& rhs) {
  return rhs + lhs;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::sub_scalar, OpInfo<T>::sub_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::mul_scalar, OpInfo<T>::mul_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(T lhs, const dpu_vector<T>& rhs) {
  return rhs * lhs;
}

template <typename T>
dpu_vector<T> operator/(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::div_scalar, OpInfo<T>::div_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

#if PIPELINE
template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(const pipeline_result<T>& other) {
  this->data_ = other.vec.data_desc_ref();
  this->size_ = other.vec.size();
  return *this;
}

template <typename T>
pipeline_result<T> dpu_vector<T>::pipeline(const std::vector<uint8_t>& ops) {
  return pipeline(ops, {});
}
#endif

#if PIPELINE
template <typename T>
std::vector<uint8_t> dpu_vector<T>::prepare_rpn(
    const std::vector<uint8_t>& ops) {
  std::vector<uint8_t> rpn_ops;
  bool is_rpn = !ops.empty() && (ops[0] >= OP_PUSH_INPUT);
  if (is_rpn) {
    rpn_ops = ops;
  } else {
    // Check if ops are already RPN (contain PUSH instructions)
    bool is_raw_rpn = false;
    for (uint8_t op : ops) {
      if (op == OP_PUSH_INPUT ||
          (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7)) {
        is_raw_rpn = true;
        break;
      }
    }

    if (is_raw_rpn) {
      rpn_ops = ops;
    } else {
      // Translate Linear -> RPN
      if (!ops.empty()) {
        rpn_ops.push_back(OP_PUSH_INPUT);
        size_t next_operand = 0;
        for (uint8_t op : ops) {
          bool is_binary = (op >= OP_ADD && op <= OP_DIV);
          if (is_binary) {
            if (next_operand < MAX_VFUSE_INPUTS) {
              rpn_ops.push_back(OP_PUSH_OPERAND_0 + next_operand);
              next_operand++;
            }
          }
          rpn_ops.push_back(op);
        }
      }
    }
  }
  return rpn_ops;
}
#endif

#if PIPELINE
template <typename T>
pipeline_result<T> dpu_vector<T>::pipeline(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "pipeline_intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  std::vector<uint8_t> rpn_ops = prepare_rpn(ops);
  std::vector<detail::VectorDescRef> operand_refs;
  std::vector<detail::VectorDescRef> absorbed_inputs;
  absorbed_inputs.push_back(this->data_desc_ref());
  for (const auto& op : operands) {
    operand_refs.push_back(op.data_desc_ref());
    absorbed_inputs.push_back(op.data_desc_ref());
  }

  // Mark the JIT-produced vector as an absorbable intermediate so a later
  // consumer can inline it instead of forcing an MRAM round-trip.
  res.data_desc_ref()->absorbed_rpn = rpn_ops;
  res.data_desc_ref()->absorbed_scalars = scalars;
  res.data_desc_ref()->absorbed_inputs = absorbed_inputs;
  res.data_desc_ref()->is_shared_intermediate = true;

  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    rpn_ops, operand_refs,
                                    OpInfo<T>::universal_pipeline, scalars);
  return res;
}
#endif

#if JIT
#include "jit.h"

template <typename T>
pipeline_result<T> dpu_vector<T>::jit(const std::vector<uint8_t>& ops) {
  return jit(ops, {});
}

template <typename T>
pipeline_result<T> dpu_vector<T>::jit(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "jit_result";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;

  std::vector<uint8_t> rpn_ops = prepare_rpn(ops);
  std::vector<detail::VectorDescRef> operand_refs;
  std::vector<detail::VectorDescRef> absorbed_inputs;
  absorbed_inputs.push_back(this->data_desc_ref());
  for (const auto& op : operands) {
    operand_refs.push_back(op.data_desc_ref());
    absorbed_inputs.push_back(op.data_desc_ref());
  }

  // Allow later consumers to inline this JIT result rather than materializing
  // it in MRAM if the queue can absorb it.
  res.data_desc_ref()->absorbed_rpn = rpn_ops;
  res.data_desc_ref()->absorbed_scalars = scalars;
  res.data_desc_ref()->absorbed_inputs = absorbed_inputs;
  res.data_desc_ref()->is_shared_intermediate = true;

  // Compiler invocation
  const char* tname;
  if (std::is_same<T, int>::value) {
    tname = "int32_t";
  } else if (std::is_same<T, uint32_t>::value) {
    tname = "uint32_t";
  } else {
    tname = typeid(T).name();
  }
  std::vector<std::pair<std::vector<uint8_t>, std::string>> kernels = {
      {rpn_ops, tname}};
  std::string binary_path = jit_compile(kernels);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE);

  e->jit_binary_path = binary_path;
  e->slice_name = "JIT Kernel";

  // Reuse the pipeline arguments structure
  e->output = res.data_desc_ref();
  e->inputs.push_back(this->data_desc_ref());
  e->inputs.insert(e->inputs.end(), operand_refs.begin(), operand_refs.end());
  e->rpn_ops = rpn_ops;
  e->scalars = scalars;
  e->kid = 0;  // JIT kernel doesn't use standard IDs
  e->pipeline_kid = 0;

  event_queue.submit(e);

  return res;
}
#endif

template <typename T>
typename dpu_vector<T>::reduction_result_t lazy_reduction_result<T>::get() {
  if (!vec.data_desc_ref()) return {};
  return reduction_cpu(vec, rid);
}

#if PIPELINE
template <typename T>
lazy_reduction_result<T> dpu_vector<T>::pipeline_reduce(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  auto& runtime = DpuRuntime::get();

  // Identify rid from last op
  assert(!ops.empty());
  uint8_t last_op = ops.back();
  KernelID rid = OpInfo<T>::sum;  // default
  switch (last_op) {
    case OP_MIN:
      rid = OpInfo<T>::min;
      break;
    case OP_MAX:
      rid = OpInfo<T>::max;
      break;
    case OP_SUM:
      rid = OpInfo<T>::sum;
      break;
    case OP_PRODUCT:
      rid = OpInfo<T>::product;
      break;
  }

  // We always allocate 8 bytes per DPU for reduction results to satisfy
  // mram_write minimum alignment and size requirements.
  size_t elems_per_dpu = (8 + sizeof(T) - 1) / sizeof(T);

  dpu_vector<T> res(runtime.num_dpus() * elems_per_dpu,
                    runtime.num_tasklets() * 8, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->is_reduction_result = true;
  res.data_desc_ref()->reduction_rid = rid;

  std::vector<detail::VectorDescRef> operand_descs;
  for (const auto& op : operands) operand_descs.push_back(op.data_desc_ref());

  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    ops, operand_descs,
                                    OpInfo<T>::universal_pipeline, scalars);

  return lazy_reduction_result<T>(std::move(res), rid);
}

template <typename T>
lazy_reduction_result<T> dpu_vector<T>::pipeline_reduce(
    const dpu_pipeline_expr<T>& expr,
    const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  return pipeline_reduce(expr.ops(), operands, scalars);
}

template <typename T>
template <typename F>
lazy_reduction_result<T> dpu_vector<T>::transform_reduce(
    F&& build, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  std::vector<dpu_expr<T>> vars;
  vars.reserve(operands.size() + 1);
  vars.push_back(dpu_expr<T>::input());
  for (size_t i = 0; i < operands.size(); ++i) {
    vars.push_back(dpu_expr<T>::operand((uint8_t)i));
  }

  dpu_expr<T> expr = build(vars);
  return pipeline_reduce(expr, operands, scalars);
}
#endif

#if PIPELINE
template <typename T>
pipeline_result<T>::operator T() {
  if (vec.data_desc().is_reduction_result) {
    return (T)lazy_reduction_result<T>(std::move(vec),
                                       vec.data_desc().reduction_rid);
  }
  // If not a reduction, return first element as a best effort scalar conversion
  return vec.to_cpu()[0];
}
#endif

// Unary operators
template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(),
                       OpInfo<T>::negate, OpInfo<T>::negate_op,
                       OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(), OpInfo<T>::abs,
                       OpInfo<T>::abs_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
lazy_reduction_result<T> sum(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::sum, OpInfo<T>::sum_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::sum);
}

template <typename T>
lazy_reduction_result<T> product(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::product, OpInfo<T>::product_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::product);
}

template <typename T>
lazy_reduction_result<T> min(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(), runtime.num_tasklets() * sizeof(size_t),
                    true);
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::min, OpInfo<T>::min_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::min);
}

template <typename T>
lazy_reduction_result<T> max(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::max, OpInfo<T>::max_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::max);
}

namespace detail {
inline uint8_t local_reduce_opcode(dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      return OP_SUM;
    case dpu_local_reduce_op::product:
      return OP_PRODUCT;
    case dpu_local_reduce_op::min:
      return OP_MIN;
    case dpu_local_reduce_op::max:
      return OP_MAX;
  }
  return OP_SUM;
}

template <typename T>
T local_reduce_identity(dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      return static_cast<T>(0);
    case dpu_local_reduce_op::product:
      return static_cast<T>(1);
    case dpu_local_reduce_op::min:
      return std::numeric_limits<T>::max();
    case dpu_local_reduce_op::max:
      return std::numeric_limits<T>::lowest();
  }
  return static_cast<T>(0);
}

template <typename T>
void local_reduce_apply(T& dst, const T& value, dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      dst += value;
      return;
    case dpu_local_reduce_op::product:
      dst *= value;
      return;
    case dpu_local_reduce_op::min:
      if (value < dst) dst = value;
      return;
    case dpu_local_reduce_op::max:
      if (value > dst) dst = value;
      return;
  }
}
}  // namespace detail

template <typename T>
dpu_local_vector<T>::dpu_local_vector(uint32_t n, std::string_view name,
                                      std::source_location loc)
    : dpu_local_vector(n, dpu_local_reduce_op::sum, name, loc) {}

template <typename T>
dpu_local_vector<T>::dpu_local_vector(uint32_t n, dpu_local_reduce_op reduce_op,
                                      std::string_view name,
                                      std::source_location loc)
    : size_(n), reduce_op_(reduce_op) {
  auto& runtime = DpuRuntime::get();
  data_ = runtime.get_allocator().allocate_local_vector(n, sizeof(T));
  data_->is_local_vector = true;
  data_->local_reduce_opcode = detail::local_reduce_opcode(reduce_op_);
  data_->type_name = typeid(T).name();
  data_->debug_name = name.data();
  data_->debug_file = loc.file_name();
  data_->debug_line = loc.line();
}

template <typename T>
vector<T> dpu_local_vector<T>::to_cpu() {
  auto& runtime = DpuRuntime::get();
  uint32_t nr_dpus = runtime.num_dpus();
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = runtime.get_logger();
  logger.lock() << "[LOCAL-TO-CPU] wait compute last_producer_id="
                << data_->last_producer_id << " size=" << size_
                << " dpus=" << nr_dpus << std::endl;
#endif
  runtime.get_event_queue().process_events(data_->last_producer_id);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[LOCAL-TO-CPU] compute complete last_producer_id="
                << data_->last_producer_id << std::endl;
#endif

  if (data_->last_producer_id == 0) {
    return std::vector<T>(size_, detail::local_reduce_identity<T>(reduce_op_));
  }

  std::vector<T> all_data(size_ * nr_dpus);
  char* cpu_buffer = reinterpret_cast<char*>(all_data.data());
  auto bound_cb = std::bind(detail::vec_xfer_from_dpu, cpu_buffer, data_);
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);
  e->inputs = {data_};
  e->host_ptr = cpu_buffer;
  e->transfer_size = all_data.size() * sizeof(T);

#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[LOCAL-TO-CPU] host transfer submit bytes="
                << e->transfer_size << std::endl;
#endif
  event_queue.submit(e);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[LOCAL-TO-CPU] host transfer wait id=" << e->id
                << std::endl;
#endif
  event_queue.process_events(e->id);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[LOCAL-TO-CPU] host transfer complete id=" << e->id
                << std::endl;
#endif

  std::vector<T> merged(size_, detail::local_reduce_identity<T>(reduce_op_));
  for (uint32_t d = 0; d < nr_dpus; d++) {
    const uint32_t base = d * size_;
    for (uint32_t i = 0; i < size_; i++) {
      detail::local_reduce_apply(merged[i], all_data[base + i], reduce_op_);
    }
  }
  return merged;
}

#if JIT
template <typename T>
detail::jit_indirect_load_expr<T> dpu_vector<T>::operator[](
    detail::jit_index_expr idx) const {
  uint8_t op_id = idx.rec.add_operand(this->data_desc_ref());
  idx.rec.rpn.push_back(OP_PUSH_INDEX);
  idx.rec.rpn.push_back(OP_LOAD_INDIRECT);
  idx.rec.rpn.push_back(op_id);
  return {*this, idx.rec};
}
#endif

#if JIT
template <typename T>
detail::jit_indirect_load_expr<T> detail::jit_indirect_load_expr<T>::operator*(
    T rhs) const {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &rhs, sizeof(T) < 4 ? sizeof(T) : 4);
  rec.rpn.push_back(OP_PUSH_SCALAR);
  rec.rpn.push_back((uint8_t)(scalar_bits & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 8) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 16) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 24) & 0xFF));
  rec.rpn.push_back(OP_MUL);
  return {*this};
}

template <typename T>
detail::jit_indirect_load_expr<T> detail::jit_indirect_load_expr<T>::operator+(
    T rhs) const {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &rhs, sizeof(T) < 4 ? sizeof(T) : 4);
  rec.rpn.push_back(OP_PUSH_SCALAR);
  rec.rpn.push_back((uint8_t)(scalar_bits & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 8) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 16) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 24) & 0xFF));
  rec.rpn.push_back(OP_ADD);
  return {*this};
}

template <typename T>
detail::jit_indirect_load_expr<T> detail::jit_indirect_load_expr<T>::operator-(
    T rhs) const {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &rhs, sizeof(T) < 4 ? sizeof(T) : 4);
  rec.rpn.push_back(OP_PUSH_SCALAR);
  rec.rpn.push_back((uint8_t)(scalar_bits & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 8) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 16) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 24) & 0xFF));
  rec.rpn.push_back(OP_SUB);
  return {*this};
}
#endif

#if JIT
template <typename T>
detail::jit_indirect_ref_expr<T> dpu_local_vector<T>::operator[](
    const detail::jit_indirect_load_expr<T>& idx) {
  return {*this, idx.rec};
}
#endif

namespace detail {
template <typename T>
void jit_indirect_ref_expr<T>::operator++(int) {
  if (vec.reduce_op() != dpu_local_reduce_op::sum) {
    throw std::logic_error(
        "dpu_local_vector::operator++ is only valid for sum reducers; use "
        ".apply() for custom local reductions");
  }
  apply(static_cast<T>(1));
}

template <typename T>
void jit_indirect_ref_expr<T>::apply(T value) {
  uint8_t local_id = rec.add_local(vec.data_desc_ref());
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &value, sizeof(T) < 4 ? sizeof(T) : 4);
  rec.rpn.push_back(OP_PUSH_SCALAR);
  rec.rpn.push_back((uint8_t)(scalar_bits & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 8) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 16) & 0xFF));
  rec.rpn.push_back((uint8_t)((scalar_bits >> 24) & 0xFF));
  rec.rpn.push_back(OP_APPLY_INDIRECT);
  rec.rpn.push_back(local_id);
  rec.rpn.push_back(detail::local_reduce_opcode(vec.reduce_op()));
}

template <typename T>
void jit_indirect_ref_expr<T>::apply(
    const detail::jit_indirect_load_expr<T>& value) {
  (void)value;
  uint8_t local_id = rec.add_local(vec.data_desc_ref());
  rec.rpn.push_back(OP_APPLY_INDIRECT);
  rec.rpn.push_back(local_id);
  rec.rpn.push_back(detail::local_reduce_opcode(vec.reduce_op()));
}
}  // namespace detail

#if JIT
template <typename T, typename F>
void dpu_jit_foreach(uint32_t n, F f) {
  detail::jit_recorder rec;
  detail::jit_index_expr idx{rec};

  // Record the loop body
  f(idx);

  for (const auto& local : rec.locals) {
    if (local && local->num_elements > MAX_LOCAL_VECTOR_SIZE) {
      throw std::logic_error(
          "dpu_local_vector exceeds MAX_LOCAL_VECTOR_SIZE for WRAM-backed JIT "
          "lowering");
    }
  }
  if (rec.locals.size() > MAX_LOCAL_SCRATCH_VECTORS) {
    throw std::logic_error(
        "too many dpu_local_vector scratch buffers for current JIT lowering");
  }

  // Submit JIT Kernel
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE);

  // Compiler invocation
  const char* tname = typeid(T).name();
  if (std::is_same<T, int32_t>::value) tname = "int32_t";
  if (std::is_same<T, float>::value) tname = "float";

  std::vector<std::pair<std::vector<uint8_t>, std::string>> kernels = {
      {rec.rpn, tname}};
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = runtime.get_logger();
  logger.lock() << "[JIT-FOREACH] compile start rpn=" << rec.rpn.size()
                << " locals=" << rec.locals.size()
                << " operands=" << rec.operands.size() << std::endl;
#endif
  std::string binary_path = jit_compile(kernels);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[JIT-FOREACH] compile end" << std::endl;
#endif

  e->jit_binary_path = binary_path;
  e->slice_name = "JIT Indirect Loop";
  e->rpn_ops = rec.rpn;

  // The generic pipeline launcher treats inputs[0] as the init vector used to
  // derive per-DPU element counts, and maps inputs[1..] to binary_operands[].
  // Indirect JIT loops index through rec.operands via OP_LOAD_INDIRECT, so
  // mirror the first operand into the init slot while keeping the full operand
  // list available as binary_operands[].
  if (!rec.operands.empty()) {
    e->inputs.push_back(rec.operands[0]);
  }
  e->inputs.insert(e->inputs.end(), rec.operands.begin(), rec.operands.end());
  // Local vectors are mapped to extra_res_offsets
  e->extra_outputs = rec.locals;

#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[JIT-FOREACH] submit start" << std::endl;
#endif
  event_queue.submit(e);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock() << "[JIT-FOREACH] submit end" << std::endl;
#endif
}
#endif
