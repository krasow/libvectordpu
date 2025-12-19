#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>
#include <type_traits>

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
  dpu_vector<T> vec(cpu_vec.size(), 0, name, loc);
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
  runtime.debug_print_dpus();
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
  dpu_vector<T> res(lhs.size());
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::add);
  return res;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size());
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::sub);
  return res;
}

// Unary operators
template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size());
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(),
                       OpInfo<T>::negate);
  return res;
}

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size());
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(), OpInfo<T>::abs);
  return res;
}

template <typename T>
T sum(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::sum);
  return reduction_cpu(buf, OpInfo<T>::sum);
}

template <typename T>
T product(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::product);
  return reduction_cpu(buf, OpInfo<T>::product);
}

template <typename T>
T min(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::min);
  return reduction_cpu(buf, OpInfo<T>::min);
}

template <typename T>
T max(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  dpu_vector<T> buf(runtime.num_dpus(),
                    runtime.num_tasklets() * sizeof(size_t));
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::max);
  return reduction_cpu(buf, OpInfo<T>::max);
}
