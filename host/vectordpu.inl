#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>
#include <type_traits>

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

template <typename T>
dpu_vector<T>::dpu_vector(uint32_t n, uint32_t reserved, std::string_view name,
                          std::source_location loc)
    : size_(n),
      reserved_(reserved),
      debug_name(name.data()),
      debug_file(loc.file_name()),
      debug_line(loc.line()) {
  auto& runtime = DpuRuntime::get();

  if (runtime.is_initialized() == false) {
    // throw std::runtime_error("DPU runtime not initialized!");
    int nr_dpus = 8;
    const char* env_val = std::getenv("NR_DPUS");
    if (env_val != nullptr) {
        nr_dpus = std::atoi(env_val);  // convert string to int
    }
    runtime.init(nr_dpus);
  }

  if (!copied) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    log_allocation(logger, typeid(T), n, debug_name, debug_file, debug_line);
#endif

    data_ =
        runtime.get_allocator().allocate_upmem_vector(n, reserved, sizeof(T));

#if ENABLE_DPU_LOGGING >= 2
    print_vector_desc(logger, data_, reserved);
#endif
  }
}

template <typename T>
dpu_vector<T>::dpu_vector(const dpu_vector& other) {
  if (this != &other) {
    data_ = other.data_;
    size_ = other.size_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    copied = true;
  }
  // #if ENABLE_DPU_LOGGING >= 2
  //   Logger& logger = DpuRuntime::get().get_logger();
  //   logger.lock() << "[dpu_vector] COPY CONSTRUCTOR at " << debug_name
  //                 << " OF SIZE " << size_ << " FROM " << debug_file << ":"
  //                 << debug_line << std::endl;
  // #endif
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(const dpu_vector& other) {
  if (this != &other) {
    data_ = other.data_;
    size_ = other.size_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    copied = true;
  }
  // #if ENABLE_DPU_LOGGING >= 2
  //   Logger& logger = DpuRuntime::get().get_logger();
  //   logger.lock() << "[dpu_vector] COPY ASSIGNMENT at " << debug_name
  //                 << " OF SIZE " << size_ << " FROM " << debug_file << ":"
  //                 << debug_line << std::endl;
  // #endif
  return *this;
}

template <typename T>
dpu_vector<T>::~dpu_vector() {
  if (copied == true) {
    // Nothing to deallocate
    return;
  }
  auto& runtime = DpuRuntime::get();

#if ENABLE_DPU_LOGGING >= 1
  if (!copied) {
    Logger& logger = DpuRuntime::get().get_logger();
    log_deallocation(logger, typeid(T), size_, debug_name, debug_file,
                     debug_line);
  }
#endif

  runtime.get_allocator().deallocate_upmem_vector(data_);
}

template <typename T>
vector<uint32_t> dpu_vector<T>::data() const {
  // data_ is vector_desc std::pair<vector<uint32_t>, vector<uint32_t>>
  // where first element is vector of pointers to DPU memory per DPU
  // and second element is vector of sizes per DPU
  return data_.first;
}

template <typename T>
uint32_t dpu_vector<T>::size() const {
  return size_;
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

inline void vec_xfer_to_dpu(char* cpu_vec, vector_desc desc,
                            uint32_t reserved) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
    element += desc.second[idx_dpu] - reserved;
  }

  uint32_t mram_location = desc.first[0];
  size_t xfer_size = desc.second[0] - reserved;

  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

inline void vec_xfer_from_dpu(char* cpu_vec, vector_desc desc,
                              uint32_t reserved) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
    element += desc.second[idx_dpu] - reserved;
  }

  uint32_t mram_location = desc.first[0];
  size_t xfer_size = desc.second[0] - reserved;
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec,
                                      std::string_view name,
                                      std::source_location loc) {
  dpu_vector<T> vec(cpu_vec.size(), 0, name, loc);
  // .data returns a std::pair<vector<uint32_t>, vector<uint32_t>>
  // the first element is vector of pointers to DPU memory per DPU
  // the second element is vector of sizes per DPU
  auto desc = vec.data_desc();

  uint32_t reserved = vec.reserved();
  char* cpu_buffer = reinterpret_cast<char*>(cpu_vec.data());
  auto bound_cb = std::bind(vec_xfer_to_dpu, cpu_buffer, desc, reserved);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::DPU_TRANSFER, bound_cb);

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
  auto desc = this->data_desc();  // pair< vector<uint32_t>, vector<uint32_t> >
  uint32_t reserved = this->reserved();

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
  auto bound_cb = std::bind(vec_xfer_from_dpu, cpu_buffer, desc, reserved);
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);

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
  dpu_set_t dpu;
  dpu_set_t& set = runtime.dpu_set();
  DPU_FOREACH(set, dpu) { DPU_ASSERT(dpu_log_read(dpu, stdout)); }
#endif
#endif

  return cpu_vec;
}

template <typename T>
void internal_launch_binop(dpu_vector<T>& res, const dpu_vector<T>& lhs,
                           const dpu_vector<T>& rhs, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_BINARY);
    args[i].num_elements = lhs.data_desc().second[i] / sizeof(T);
    args[i].size_type = sizeof(T);
    args[i].binary.lhs_offset = reinterpret_cast<uint32_t>(lhs.data()[i]);
    args[i].binary.rhs_offset = reinterpret_cast<uint32_t>(rhs.data()[i]);
    args[i].binary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
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

template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs,
                           KernelID kernel_id) {
  dpu_vector<T> res(lhs.size());
  launch_binop<T>(res, lhs, rhs, kernel_id);
  return res;
}

template <typename T>
void launch_binop(dpu_vector<T>& res, const dpu_vector<T>& lhs,
                  const dpu_vector<T>& rhs, KernelID kernel_id) {
  assert(lhs.size() == rhs.size());

  auto bound_cb = std::bind(internal_launch_binop<T>, res, lhs, rhs, kernel_id);
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = make_result_descriptor(res);
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

template <typename T>
void internal_launch_unary(dpu_vector<T>& res, const dpu_vector<T>& a,
                           KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_UNARY);
    args[i].num_elements = a.data_desc().second[i] / sizeof(T);
    args[i].size_type = sizeof(T);
    args[i].unary.rhs_offset = reinterpret_cast<uint32_t>(a.data()[i]);
    args[i].unary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
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

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a, KernelID kernel_id) {
  dpu_vector<T> res(a.size());
  launch_unary<T>(res, a, kernel_id);
  return res;
}

template <typename T>
void launch_unary(dpu_vector<T>& res, const dpu_vector<T>& rhs,
                  KernelID kernel_id) {
  auto bound_cb = std::bind(internal_launch_unary<T>, res, rhs, kernel_id);
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = make_result_descriptor(res);
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

template <typename T>
void internal_launch_reduction(dpu_vector<T>& res, const dpu_vector<T>& a,
                               KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].ktype = static_cast<uint8_t>(KERNEL_REDUCTION);
    args[i].num_elements = a.data_desc().second[i] / sizeof(T);
    args[i].size_type = sizeof(T);
    args[i].reduction.rhs_offset = reinterpret_cast<uint32_t>(a.data()[i]);
    args[i].reduction.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
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

template <typename T>
T launch_reduction(const dpu_vector<T>& a, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> res(runtime.num_dpus(),
                    NR_TASKLETS * sizeof(size_t));  // each dpu returns a single

  auto bound_cb = std::bind(internal_launch_reduction<T>, res, a, kernel_id);
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);

  e->res = make_result_descriptor(res);
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] type=COMPUTE kernel="
                << kernel_id_to_string(kernel_id) << std::endl;
#endif

  vector<T> res_cpu = res.to_cpu();

  assert(res_cpu.size() % runtime.num_dpus() == 0);

  size_t stride = res_cpu.size() / runtime.num_dpus();
  // initialize accumulator with the first partial result
  T acc = res_cpu[0];

  // reduce over the remaining DPUs
  reduction_op_t op = get_reduction_op(kernel_id);
  for (size_t i = stride; i < res_cpu.size(); i += stride) {
    T x = res_cpu[i];
    switch (op) {
      case REDUCTION_OP_SUM:
        acc += x;
        break;
      case REDUCTION_OP_PRODUCT:
        acc *= x;
        break;
      case REDUCTION_OP_MAX:
        acc = (x > acc) ? x : acc;
        break;
      case REDUCTION_OP_MIN:
        acc = (x < acc) ? x : acc;
        break;
      default:
        assert(false && "Unknown reduction operation");
    }
  }
  return acc;
}