#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>
#include <type_traits>

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

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
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
      copied(true) {}

template <typename T>
dpu_vector<T>::dpu_vector(dpu_vector&& other) noexcept
    : data_(std::move(other.data_)),
      size_(other.size_),
      reserved_(other.reserved_),
      debug_name(other.debug_name),
      debug_file(other.debug_file),
      debug_line(other.debug_line),
      copied(false) {
  other.copied = true;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(const dpu_vector& other) {
  if (this != &other) {
    if (!copied && data_) {
      auto& runtime = DpuRuntime::get();
#if ENABLE_DPU_LOGGING >= 1
      Logger& logger = runtime.get_logger();
      log_deallocation(logger, typeid(T), size_, debug_name, debug_file,
                       debug_line);
#endif
      runtime.get_allocator().deallocate_upmem_vector(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    copied = true;
  }
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(dpu_vector&& other) noexcept {
  if (this != &other) {
    if (!copied && data_) {
      auto& runtime = DpuRuntime::get();
#if ENABLE_DPU_LOGGING >= 1
      Logger& logger = runtime.get_logger();
      log_deallocation(logger, typeid(T), size_, debug_name, debug_file,
                       debug_line);
#endif
      runtime.get_allocator().deallocate_upmem_vector(data_);
    }
    data_ = std::move(other.data_);
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    copied = false;
    other.copied = true;
  }
  return *this;
}

template <typename T>
dpu_vector<T>::~dpu_vector() {
  if (!copied && data_) {
    auto& runtime = DpuRuntime::get();
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = runtime.get_logger();
    log_deallocation(logger, typeid(T), size_, debug_name, debug_file,
                     debug_line);
#endif
    runtime.get_allocator().deallocate_upmem_vector(data_);
  }
}

template <typename T>
void dpu_vector<T>::add_fence() {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::FENCE);

  event_queue.submit(e);
  event_queue.process_events(e->id);
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec,
                                      std::string_view name,
                                      std::source_location loc) {
  dpu_vector<T> vec(cpu_vec.size(), 0, false, name, loc);
  // .data returns a std::pair<vector<uint32_t>, vector<uint32_t>>
  // the first element is vector of pointers to DPU memory per DPU
  // the second element is vector of sizes per DPU
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

  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=HOST_TRANSFER size=" << cpu_vec.size()
                << std::endl;
#endif

// Auto-fence after DPU->HOST transfer if enabled
#if ENABLE_AUTO_FENCING == 1
  event_queue.process_events(e->id);
// need the event to be completed before reading printf output
#if ENABLE_DPU_PRINTING == 1
  // read and print DPU logs to host stdout
  runtime.debug_read_dpu_log();
#endif
#endif

  return cpu_vec;
}

template <typename T>
T reduction_cpu(dpu_vector<T>& da, KernelID kernel_id) {
  // block and send to cpu
  auto a = da.to_cpu();

  auto& runtime = DpuRuntime::get();
  assert(a.size() % runtime.num_dpus() == 0);
  size_t stride = a.size() / runtime.num_dpus();
  // initialize accumulator with the first partial result
  T acc = a[0];

  // reduce over the remaining DPUs
  auto op = kernel_infos[kernel_id].op;
  for (size_t i = stride; i < a.size(); i += stride) {
    T x = a[i];
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
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               static_cast<uint32_t>(scalar),
                               OpInfo<T>::add_scalar, OpInfo<T>::add_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator-=(T scalar) {
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               static_cast<uint32_t>(scalar),
                               OpInfo<T>::sub_scalar, OpInfo<T>::sub_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator*=(T scalar) {
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               static_cast<uint32_t>(scalar),
                               OpInfo<T>::mul_scalar, OpInfo<T>::mul_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator/=(T scalar) {
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               static_cast<uint32_t>(scalar),
                               OpInfo<T>::div_scalar, OpInfo<T>::div_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator>>=(T scalar) {
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               static_cast<uint32_t>(scalar),
                               OpInfo<T>::asr_scalar, OpInfo<T>::asr_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::operator-() const {
  dpu_vector<T> res(this->size(), 0, true);
  detail::launch_unary(res.data_desc_ref(), this->data_desc_ref(),
                       OpInfo<T>::negate, OpInfo<T>::negate_op,
                       OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator>>(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary_scalar(
      res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint32_t>(rhs),
      OpInfo<T>::asr_scalar, OpInfo<T>::asr_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary_scalar(
      res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint32_t>(rhs),
      OpInfo<T>::add_scalar, OpInfo<T>::add_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(T lhs, const dpu_vector<T>& rhs) {
  return rhs + lhs;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary_scalar(
      res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint32_t>(rhs),
      OpInfo<T>::sub_scalar, OpInfo<T>::sub_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary_scalar(
      res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint32_t>(rhs),
      OpInfo<T>::mul_scalar, OpInfo<T>::mul_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(T lhs, const dpu_vector<T>& rhs) {
  return rhs * lhs;
}

template <typename T>
dpu_vector<T> operator/(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary_scalar(
      res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint32_t>(rhs),
      OpInfo<T>::div_scalar, OpInfo<T>::div_op, OpInfo<T>::universal_pipeline);
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
pipeline_result<T> dpu_vector<T>::pipeline(
    const std::vector<uint8_t>& ops,
    const std::vector<dpu_vector<T>>& operands) {
  dpu_vector<T> res(this->size(), 0, true);
  std::vector<uint8_t> rpn_ops;
  // Check if it already looks like RPN (starts with a PUSH)
  // PUSH_INPUT=11, PUSH_OPERAND_X=12-19
  bool is_rpn = !ops.empty() && (ops[0] >= OP_PUSH_INPUT);

  if (is_rpn) {
    rpn_ops = ops;
  } else {
    // Translate Linear -> RPN
    if (!ops.empty()) {
      rpn_ops.push_back(OP_PUSH_INPUT);  // OP_PUSH_INPUT
      size_t next_operand = 0;
      for (uint8_t op : ops) {
        // ADD=3, SUB=4, MUL=5, DIV=6
        bool is_binary = (op >= OP_ADD && op <= OP_DIV);
        if (is_binary) {
          if (next_operand < MAX_PIPELINE_OPERANDS) {
            rpn_ops.push_back(OP_PUSH_OPERAND_0 +
                              next_operand);  // OP_PUSH_OPERAND_0..7
            next_operand++;
          }
        }
        rpn_ops.push_back(op);
      }
    }
  }

  std::vector<detail::VectorDescRef> operand_refs;
  for (const auto& op : operands) {
    operand_refs.push_back(op.data_desc_ref());
  }

  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    rpn_ops, operand_refs,
                                    OpInfo<T>::universal_pipeline);
  return res;
}
#endif

#if PIPELINE
template <typename T>
T dpu_vector<T>::pipeline_reduce(const std::vector<uint8_t>& ops,
                                 const std::vector<dpu_vector<T>>& operands) {
  // A reduction pipeline must return a scalar.
  // We reuse pipeline() but return the aggregated result.
  dpu_vector<T> res = pipeline(ops, operands);

  // The last op defines the reduction type
  assert(!ops.empty());
  uint8_t last_op = ops.back();

  // Map opcode back to a standard KernelID for reduction_cpu
  // Standard reduction OPs are MIN, MAX, SUM, PRODUCT
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
    default:
      // rid remains OpInfo<T>::sum (the default)
      break;
  }

  return reduction_cpu(res, rid);
}
#endif

#if PIPELINE
template <typename T>
pipeline_result<T>::operator T() {
  if (vec.data_desc().is_reduction_result) {
    return reduction_cpu(const_cast<dpu_vector<T>&>(vec),
                         vec.data_desc().reduction_rid);
  }
  // If not a reduction, return first element as a best effort scalar conversion
  return vec.to_cpu()[0];
}
#endif

// Unary operators
template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size());
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(),
                       OpInfo<T>::negate, OpInfo<T>::negate_op,
                       OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size());
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(), OpInfo<T>::abs,
                       OpInfo<T>::abs_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
T sum(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::sum, OpInfo<T>::sum_op,
                           OpInfo<T>::universal_pipeline);
  return reduction_cpu(buf, OpInfo<T>::sum);
}

template <typename T>
T product(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::product, OpInfo<T>::product_op,
                           OpInfo<T>::universal_pipeline);
  return reduction_cpu(buf, OpInfo<T>::product);
}

template <typename T>
T min(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(), runtime.num_tasklets() * sizeof(size_t),
                    true);
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::min, OpInfo<T>::min_op,
                           OpInfo<T>::universal_pipeline);
  return reduction_cpu(buf, OpInfo<T>::min);
}

template <typename T>
T max(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::max, OpInfo<T>::max_op,
                           OpInfo<T>::universal_pipeline);
  return reduction_cpu(buf, OpInfo<T>::max);
}
